"""Dataset adapter registry and built-in window samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.config import EGODEX_ROOT
from src.data.camera import (
    CAMERA_EXTRINSICS_SEMANTICS,
    TARGET_CAMERA_EQUATION,
    TARGET_CAMERA_EQUATION_NO_FRAME,
    TARGET_CAMERA_REFERENCE,
    TARGET_CAMERA_SEMANTICS,
    camera_flow_from_extrinsics,
    world_points_to_first_camera,
)
from .contract import WindowSample


class DatasetAdapterError(ValueError):
    pass


@dataclass
class EgoDexAdapter:
    """Adapter-owned EgoDex episode/split/window selection.

    The adapter exposes only the public window contract. ``split`` is interpreted
    here, never by the training loop. Conditions are explicit missing tensors;
    EgoDex 3D targets do not enter the 2D keypoint condition field.
    """

    root: Any = EGODEX_ROOT
    split: str = "test"
    episode: str | None = None
    window_start: int | None = None

    def __post_init__(self) -> None:
        from .egodex import episode_ids

        self.root = type(EGODEX_ROOT)(self.root)
        self._episodes = episode_ids(self.split, self.root)
        if self.episode is not None:
            if self.episode not in self._episodes:
                raise DatasetAdapterError(
                    f"episode {self.episode!r} is not available in adapter split {self.split!r}"
                )
            self._episodes = [self.episode]

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": "egodex",
            "split": self.split,
            "episodes": len(self._episodes),
            "episode": self.episode,
            "window_start": self.window_start,
            "coordinate_contract": "canonical first-model-camera coordinates for targets; source units",
            "raw_coordinate_frame": "episode_world",
            "camera_extrinsics_semantics": CAMERA_EXTRINSICS_SEMANTICS,
            "camera_extrinsics_camera_frame": "source_camera",
            "target_camera_semantics": TARGET_CAMERA_SEMANTICS,
            "target_camera_equation": TARGET_CAMERA_EQUATION_NO_FRAME,
            "target_camera_reference_frame": TARGET_CAMERA_REFERENCE,
            "target_camera_camera_frame": "source_camera",
            "joint_unit": "source",
            "intrinsics_frame": "original_video_pixels",
            "image_transform": "none",
            "joint_order": "public_16",
        }

    def _window(self, episode_id: str, window: int, start: int | None):
        from .egodex import EgoDexDataset

        full_dataset = EgoDexDataset(episode_id, root=self.root)
        if window < 1 or window > full_dataset.num_frames:
            raise DatasetAdapterError(
                f"window {window} is invalid for {episode_id} ({full_dataset.num_frames} frames)"
            )
        if start is None:
            start = int(np.random.default_rng().integers(0, full_dataset.num_frames - window + 1))
        if start < 0 or start + window > full_dataset.num_frames:
            raise DatasetAdapterError(f"window_start {start} is invalid for {episode_id}")
        from .egodex import EgoDexDataset
        return EgoDexDataset(episode_id, frames=list(range(start, start + window)), root=self.root), list(range(start, start + window))

    def sample(self, *, seed: int, batch_size: int, window: int) -> WindowSample:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(seed)
        samples = []
        images = []
        metadata = []
        for _ in range(batch_size):
            episode_id = self._episodes[int(rng.integers(len(self._episodes)))]
            start = self.window_start
            if start is None:
                # Pass a deterministic adapter-owned start to the dataset.
                start = int(rng.integers(0, self._frame_count(episode_id) - window + 1))
            dataset, frames = self._window(episode_id, window, start)
            joints, valid, confidence = dataset.joints_with_confidence()
            extrinsics, intrinsics, intrinsic_fallback = dataset.camera()
            samples.append((joints, valid, confidence, extrinsics, intrinsics, intrinsic_fallback))
            images.append(torch.stack([
                torch.from_numpy(dataset.image(frame)).permute(2, 0, 1).float() / 255.0
                for frame in frames
            ]))
            metadata.append({"episode": episode_id, "frames": frames, "intrinsic_fallback": intrinsic_fallback})
        target = torch.from_numpy(np.stack([item[0] for item in samples])).float()
        valid = torch.from_numpy(np.stack([item[1] for item in samples])).bool()
        images = torch.stack(images)
        confidence_values = [item[2] for item in samples]
        confidence = None if any(value is None for value in confidence_values) else torch.from_numpy(np.stack(confidence_values)).float()
        extrinsics = torch.from_numpy(np.stack([item[3] for item in samples])).float()
        intrinsics = torch.from_numpy(np.stack([item[4] for item in samples])).float()
        target = torch.stack([
            world_points_to_first_camera(points, pose[0])
            for points, pose in zip(target, extrinsics)
        ])
        camera_flow = camera_flow_from_extrinsics(extrinsics)
        return WindowSample(
            hamer_tokens=torch.zeros(batch_size, 2, window, 1024),
            hamer_valid=torch.zeros(batch_size, 2, window, dtype=torch.bool),
            vggt_tokens=torch.zeros(batch_size, window, 17, 2048),
            vggt_valid=torch.zeros(batch_size, window, dtype=torch.bool),
            keypoints=torch.zeros(batch_size, 2, window, 21, 3),
            bbox=torch.zeros(batch_size, 2, window, 4),
            bbox_valid=torch.zeros(batch_size, 2, window, dtype=torch.bool),
            images=images,
            camera_extrinsics=extrinsics,
            camera_intrinsics=intrinsics,
            camera_valid=torch.ones(batch_size, window, dtype=torch.bool),
            hand_joints3d=target,
            target_joints3d=target,
            joint_valid=valid,
            joint_confidence=confidence,
            target_camera=camera_flow,
            metadata={
                "adapter": self.describe(),
                "samples": metadata,
                "raw_coordinate_frame": "episode_world",
                "camera_extrinsics_semantics": CAMERA_EXTRINSICS_SEMANTICS,
                "camera_extrinsics_camera_frame": "source_camera",
                "target_camera_semantics": TARGET_CAMERA_SEMANTICS,
                "target_camera_equation": TARGET_CAMERA_EQUATION_NO_FRAME,
                "target_camera_reference_frame": TARGET_CAMERA_REFERENCE,
                "target_camera_camera_frame": "source_camera",
                "hand_joints_frame": TARGET_CAMERA_REFERENCE,
                "joint_unit": "source",
                "intrinsics_frame": "original_video_pixels",
                "image_transform": "none",
                "intrinsic_fallback": any(item[5] for item in samples),
            },
        )

    def _frame_count(self, episode_id: str) -> int:
        from .egodex import EgoDexDataset

        return EgoDexDataset(episode_id, root=self.root).num_frames


@dataclass
class Hot3DAdapter:
    """Thin adapter around the existing HOT3D sampler.

    Split/clip selection is intentionally owned by this adapter.
    """

    clip_ids: list[str] | None = None
    split: str = "train"
    clip_id: str | None = None
    window: int = 64
    hamer_cache: Any = None
    handbox_cache: Any = None
    vggt: Any = None
    rectifier: Any = None
    window_starts: list[int | None] | None = None
    window_start: int | None = None

    def __post_init__(self) -> None:
        if self.window_starts is None and self.window_start is not None:
            self.window_starts = [self.window_start]
        if self.clip_ids is None:
            from src.config import DATA_ROOT
            from src.data.hot3d.splits import partition_clips
            import json
            official = json.loads((DATA_ROOT / "clip_splits.json").read_text())
            mounted = [f"{clip_id:06d}" for clip_id in official["train"]["Aria"]]
            train, test = partition_clips(mounted, train_count=1272, seed=2026)
            selected = train if self.split == "train" else test if self.split == "test" else None
            if selected is None:
                raise DatasetAdapterError(f"unknown HOT3D adapter split: {self.split}")
            self.clip_ids = [self.clip_id] if self.clip_id is not None else selected
        elif self.clip_id is not None:
            self.clip_ids = [self.clip_id]

    def describe(self) -> dict[str, Any]:
        return {
            "dataset": "hot3d",
            "split": self.split,
            "clips": len(self.clip_ids),
            "window": self.window,
            "coordinate_contract": "canonical first-model-camera coordinates for targets; m",
            "raw_coordinate_frame": "dataset_world",
            "camera_extrinsics_semantics": CAMERA_EXTRINSICS_SEMANTICS,
            "camera_extrinsics_camera_frame": "source_camera",
            "target_camera_semantics": TARGET_CAMERA_SEMANTICS,
            "target_camera_equation": TARGET_CAMERA_EQUATION,
            "target_camera_reference_frame": TARGET_CAMERA_REFERENCE,
            "target_camera_camera_frame": "upright_camera",
            "hand_joints_frame": TARGET_CAMERA_REFERENCE,
            "joint_unit": "m",
            "intrinsics_frame": "rectified_upright_pixels",
            "image_transform": "fisheye_rectification_then_clockwise_90",
        }

    def sample(self, *, seed: int, batch_size: int, window: int) -> WindowSample:
        from src.train.sample import sample_batch

        starts = self.window_starts or [None] * batch_size
        if len(starts) == 1 and batch_size > 1:
            starts = starts * batch_size
        if len(starts) != batch_size:
            raise DatasetAdapterError("window_starts must match adapter batch_size")
        sample = sample_batch(
            clip_ids=self.clip_ids,
            window=window,
            hamer_cache=self.hamer_cache,
            handbox_cache=self.handbox_cache,
            vggt=self.vggt,
            rectifier=self.rectifier,
            seeds=[seed + i for i in range(batch_size)],
            window_starts=starts,
        )
        return WindowSample(
            hamer_tokens=sample.hamer_tokens,
            hamer_valid=sample.hamer_valid,
            vggt_tokens=sample.vggt_tokens,
            vggt_valid=torch.ones(sample.vggt_tokens.shape[:2], dtype=torch.bool),
            keypoints=sample.keypoints,
            bbox=sample.bbox,
            bbox_valid=sample.bbox_valid,
            images=sample.images,
            target_mano=sample.target_mano,
            target_camera=sample.target_camera,
            camera_extrinsics=sample.camera_extrinsics,
            camera_intrinsics=sample.camera_intrinsics,
            camera_valid=sample.camera_valid,
            hand_joints3d=sample.hand_joints3d,
            target_joints3d=sample.hand_joints3d,
            joint_valid=sample.joint_valid,
            joint_confidence=sample.joint_confidence,
            metadata={
                "adapter": self.describe(),
                "raw_coordinate_frame": "dataset_world",
                "camera_extrinsics_semantics": CAMERA_EXTRINSICS_SEMANTICS,
                "camera_extrinsics_camera_frame": "source_camera",
                "target_camera_semantics": TARGET_CAMERA_SEMANTICS,
                "target_camera_equation": TARGET_CAMERA_EQUATION,
                "target_camera_reference_frame": TARGET_CAMERA_REFERENCE,
                "target_camera_camera_frame": "upright_camera",
                "hand_joints_frame": TARGET_CAMERA_REFERENCE,
                "joint_unit": "m",
                "image_transform": "fisheye_rectification_then_clockwise_90",
                "intrinsics_frame": "rectified_upright_pixels",
            },
        )


def make_adapter(name: str, **config) -> Any:
    """Create an adapter; dataset-specific split semantics stay inside it."""
    if name == "egodex":
        return EgoDexAdapter(**config)
    if name == "hot3d":
        return Hot3DAdapter(**config)
    raise ValueError(f"unknown dataset adapter: {name}")
