"""Training data sampling: window selection, conditions, and flow targets.

HaMeR (per-frame, cached) and VGGT (whole-window, uncached) run sequentially.
MANO and camera targets live in the first model-camera frame after the
dataset's clockwise-90 upright transform. Raw camera extrinsics remain in
 their source ``world_from_camera`` convention for metadata and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.data.camera import world_points_to_first_camera
from src.data.hot3d import frame_ids
from src.data.hot3d.dataset import Hot3DDataset
from src.geometry.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    quaternion_wxyz_to_matrix,
)
from src.model.flow.state import CameraFlowState, ManoFlowState

_PUBLIC_FROM_MANO = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)

# Clockwise-90 rotation applied by the dataset rectification.
_FRAME_ROTATION = torch.tensor(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)
_FRAME_TRANSFORM = torch.eye(4)
_FRAME_TRANSFORM[:3, :3] = _FRAME_ROTATION
_HANDS = ("left", "right")


@dataclass
class TrainSample:
    """A single training window with conditions and flow targets.

    Every field carries a leading batch dimension (``B=1`` for a single
    window), matching the ``[B,...]`` convention expected downstream.
    """

    hamer_tokens: Tensor  # [B,2,T,1024]
    hamer_valid: Tensor  # bool [B,2,T]
    vggt_tokens: Tensor  # [B,T,17,2048]
    keypoints: Tensor  # [B,2,T,21,3]
    bbox: Tensor  # [B,2,T,4]
    bbox_valid: Tensor  # bool [B,2,T]
    images: Tensor  # [B,T,3,H,W]
    target_mano: ManoFlowState
    target_camera: CameraFlowState
    camera_extrinsics: Tensor
    camera_intrinsics: Tensor
    camera_valid: Tensor
    hand_joints3d: Tensor
    joint_valid: Tensor
    joint_confidence: Tensor | None = None


def _se3(quat_wxyz: Tensor, transl: Tensor) -> Tensor:
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(quat_wxyz)
    matrix[:3, 3] = transl
    return matrix


def _mano_target(dataset: Hot3DDataset, rectifier) -> ManoFlowState:
    """MANO flow target from ``thetas`` (PCA) and ``wrist_xform``.

    ``wrist_xform[:3]`` is the global axis-angle rotation; ``wrist_xform[3:]``
    is the world translation. Both are transformed into the first-camera
    upright frame. Local rotations (thetas) are wrist-local and unchanged.
    """
    time = len(dataset.frames)
    first_pose = dataset.camera_pose(dataset.frames[0])
    first_from_world = torch.inverse(
        _se3(
            torch.tensor(first_pose["quaternion_wxyz"], dtype=torch.float32),
            torch.tensor(first_pose["translation_xyz"], dtype=torch.float32),
        )
    )
    state_transform = _FRAME_TRANSFORM @ first_from_world

    translation = torch.zeros(2, time, 3)
    root_rotation = torch.zeros(2, time, 6)
    local_rotation = torch.zeros(2, time, 15, 6)

    for hand_index, hand in enumerate(_HANDS):
        components, mean = rectifier.pca_components(hand)
        for frame_index, frame in enumerate(dataset.frames):
            mano_pose = dataset.hands(frame).get(hand, {}).get("mano_pose")
            if mano_pose is None:
                continue
            thetas = torch.tensor(mano_pose["thetas"], dtype=torch.float32)
            wrist = torch.tensor(mano_pose["wrist_xform"], dtype=torch.float32)

            local_axis = (thetas @ components + mean).reshape(15, 3)
            local_rotation[hand_index, frame_index] = matrix_to_rotation_6d(
                axis_angle_to_matrix(local_axis)
            )

            root_world = axis_angle_to_matrix(wrist[:3])
            root_rotation[hand_index, frame_index] = matrix_to_rotation_6d(
                state_transform[:3, :3] @ root_world
            )

            # World-space joints, then transform to the upright frame. Because
            # smplx applies a non-zero pose_mean, the root/translation do not
            # transform as a rigid body under SE(3); recover the upright
            # translation from the transformed joints via a zero-translation
            # forward, as the reference pipeline does.
            source_joints = rectifier.mano_joints(
                hand,
                wrist[:3].unsqueeze(0),
                wrist[3:].unsqueeze(0),
                local_axis.unsqueeze(0),
            )[0]
            desired = (state_transform[:3, :3] @ source_joints.T).T + state_transform[:3, 3]
            root_upright = state_transform[:3, :3] @ root_world
            zero_joints = rectifier.mano_joints(
                hand,
                matrix_to_axis_angle(root_upright).unsqueeze(0),
                torch.zeros(1, 3),
                local_axis.unsqueeze(0),
            )[0]
            translation[hand_index, frame_index] = (desired - zero_joints).mean(dim=0)

    return ManoFlowState(
        translation=translation.unsqueeze(0),
        root_rotation=root_rotation.unsqueeze(0),
        local_rotation=local_rotation.unsqueeze(0),
    )


@dataclass
class _PreparedWindow:
    hamer_tokens: Tensor
    hamer_valid: Tensor
    keypoints: Tensor
    bbox: Tensor
    bbox_valid: Tensor
    target_mano: ManoFlowState
    target_camera: CameraFlowState
    images: Tensor
    camera_extrinsics: Tensor
    camera_intrinsics: Tensor
    camera_valid: Tensor
    hand_joints3d: Tensor
    joint_valid: Tensor
    joint_confidence: Tensor | None


def _prepare_window(
    *,
    clip_ids,
    window: int,
    hamer_cache,
    handbox_cache,
    rectifier,
    seed: int,
    window_start: int | None = None,
) -> _PreparedWindow:
    rng = np.random.default_rng(seed)
    clip = str(rng.choice(clip_ids))
    all_frames = frame_ids(clip)
    if window_start is None:
        start = int(rng.integers(0, len(all_frames) - window + 1))
    else:
        if window_start < 0 or window_start + window > len(all_frames):
            raise ValueError("window_start must select a complete window")
        start = int(window_start)
    frames = all_frames[start : start + window]
    cache_root = getattr(hamer_cache, "cache_root", None)
    image_cache_root = None if not isinstance(cache_root, Path) else cache_root / "rectified-v1"
    dataset = Hot3DDataset(
        clip,
        frames,
        rectifier,
        image_cache_root=image_cache_root,
    )
    for frame in frames:
        dataset.image(frame)
        dataset.hands(frame)
        dataset.camera_pose(frame)
    if any(
        dataset.hands(frame).get(hand, {}).get("mano_pose") is None
        for frame in frames
        for hand in _HANDS
    ):
        raise ValueError("training windows require complete MANO annotations for both hands")

    if hasattr(handbox_cache, "get_detections"):
        keypoints_2d, boxes, bbox_valid_2d, _ = handbox_cache.get_detections(dataset)
    else:
        boxes, bbox_valid_2d, records = handbox_cache.get_boxes(dataset)
        if records and all(record is not None for record in records):
            keypoints_2d = torch.stack(
                [
                    torch.as_tensor(record.get("keypoints", torch.zeros(2, 21, 3)))
                    for record in records
                ],
                dim=1,
            )
        else:
            keypoints_2d = torch.zeros(2, len(dataset.frames), 21, 3)

    hamer_tokens, hamer_valid = hamer_cache.get_tokens(
        dataset, boxes, bbox_valid_2d
    )
    bbox = torch.zeros(1, 2, window, 4)
    bbox_valid = torch.zeros(1, 2, window, dtype=torch.bool)
    bbox[0] = boxes / rectifier.output_size
    bbox_valid[0] = bbox_valid_2d
    keypoints = keypoints_2d.unsqueeze(0).float()
    keypoints[..., :2] = keypoints[..., :2] / rectifier.output_size
    images = torch.stack(
        [
            torch.from_numpy(dataset.image(frame)).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ]
    )
    target_mano, target_camera = dataset.targets(rectifier)
    raw_extrinsics = []
    raw_intrinsics = []
    for frame in frames:
        pose = dataset.camera_pose(frame)
        raw_extrinsics.append(_se3(
            torch.tensor(pose["quaternion_wxyz"], dtype=torch.float32),
            torch.tensor(pose["translation_xyz"], dtype=torch.float32),
        ))
        _, _, intrinsic = dataset._calibration(frame)
        raw_intrinsics.append(torch.as_tensor(intrinsic, dtype=torch.float32))
    mano_joints = rectifier.mano_joints
    joints = torch.zeros(2, window, 16, 3)
    joint_valid = torch.zeros(2, window, 16, dtype=torch.bool)
    for hi, hand in enumerate(_HANDS):
        for ti, frame in enumerate(frames):
            pose = dataset.hands(frame).get(hand, {}).get("mano_pose")
            if pose is None:
                continue
            wrist = torch.tensor(pose["wrist_xform"], dtype=torch.float32)
            components, mean = rectifier.pca_components(hand)
            local = (torch.tensor(pose["thetas"], dtype=torch.float32) @ components + mean).reshape(15, 3)
            decoded = mano_joints(hand, wrist[:3].unsqueeze(0), wrist[3:].unsqueeze(0), local.unsqueeze(0))[0]
            joints[hi, ti] = decoded[list(_PUBLIC_FROM_MANO)]
            joint_valid[hi, ti] = True
    joints = world_points_to_first_camera(joints, raw_extrinsics[0], _FRAME_TRANSFORM)
    return _PreparedWindow(
        hamer_tokens=hamer_tokens.unsqueeze(0),
        hamer_valid=hamer_valid.unsqueeze(0),
        keypoints=keypoints,
        bbox=bbox,
        bbox_valid=bbox_valid,
        target_mano=target_mano,
        target_camera=target_camera,
        images=images,
        camera_extrinsics=torch.stack(raw_extrinsics),
        camera_intrinsics=torch.stack(raw_intrinsics),
        camera_valid=torch.ones(window, dtype=torch.bool),
        hand_joints3d=joints,
        joint_valid=joint_valid,
        joint_confidence=None,
    )


def _finish_window(prepared: _PreparedWindow, vggt_tokens: Tensor) -> TrainSample:
    if vggt_tokens.ndim == 3:
        vggt_tokens = vggt_tokens.unsqueeze(0)
    return TrainSample(
        hamer_tokens=prepared.hamer_tokens,
        hamer_valid=prepared.hamer_valid,
        vggt_tokens=vggt_tokens,
        keypoints=prepared.keypoints,
        bbox=prepared.bbox,
        bbox_valid=prepared.bbox_valid,
        images=prepared.images.unsqueeze(0),
        target_mano=prepared.target_mano,
        target_camera=prepared.target_camera,
        camera_extrinsics=prepared.camera_extrinsics.unsqueeze(0),
        camera_intrinsics=prepared.camera_intrinsics.unsqueeze(0),
        camera_valid=prepared.camera_valid.unsqueeze(0),
        hand_joints3d=prepared.hand_joints3d.unsqueeze(0),
        joint_valid=prepared.joint_valid.unsqueeze(0),
        joint_confidence=prepared.joint_confidence,
    )


def _run_vggt(vggt, images: Tensor) -> Tensor:
    if hasattr(vggt, "get_tokens"):
        raise RuntimeError("cached VGGT extraction is not supported in batched sampling")
    device = next(vggt.parameters()).device if hasattr(vggt, "parameters") else images.device
    tokens = vggt(images.to(device))
    return tokens.clone() if tokens.is_inference() else tokens


def sample_window(
    *,
    clip_ids,
    window,
    hamer_cache,
    handbox_cache,
    vggt,
    rectifier,
    seed,
    window_start: int | None = None,
) -> TrainSample:
    """Sample one consecutive-frame window and its conditions and targets."""
    prepared = _prepare_window(
        clip_ids=clip_ids,
        window=window,
        hamer_cache=hamer_cache,
        handbox_cache=handbox_cache,
        rectifier=rectifier,
        seed=seed,
        window_start=window_start,
    )
    if hasattr(vggt, "get_tokens"):
        raise RuntimeError("sample_window with a cached VGGT object requires a dataset-aware cache")
    return _finish_window(prepared, _run_vggt(vggt, prepared.images.unsqueeze(0)))


def _prepare_windows_batch(
    *,
    clip_ids,
    window: int,
    hamer_cache,
    handbox_cache,
    rectifier,
    seeds: list[int],
    window_starts: list[int | None],
) -> list[_PreparedWindow]:
    datasets = []
    for seed, window_start in zip(seeds, window_starts):
        rng = np.random.default_rng(seed)
        clip = str(rng.choice(clip_ids))
        all_frames = frame_ids(clip)
        start = (
            int(rng.integers(0, len(all_frames) - window + 1))
            if window_start is None
            else int(window_start)
        )
        if start < 0 or start + window > len(all_frames):
            raise ValueError("window_start must select a complete window")
        frames = all_frames[start : start + window]
        cache_root = getattr(hamer_cache, "cache_root", None)
        image_cache_root = None if not isinstance(cache_root, Path) else cache_root / "rectified-v1"
        dataset = Hot3DDataset(
            clip,
            frames,
            rectifier,
            image_cache_root=image_cache_root,
        )
        for frame in frames:
            dataset.image(frame)
            dataset.hands(frame)
            dataset.camera_pose(frame)
        if any(
            dataset.hands(frame).get(hand, {}).get("mano_pose") is None
            for frame in frames
            for hand in _HANDS
        ):
            raise ValueError("training windows require complete MANO annotations for both hands")
        datasets.append(dataset)

    if hasattr(handbox_cache, "get_detections_batch"):
        detections = handbox_cache.get_detections_batch(datasets)
    else:
        detections = [handbox_cache.get_detections(dataset) for dataset in datasets]
    boxes_list = [item[1] for item in detections]
    valid_list = [item[2] for item in detections]
    if hasattr(hamer_cache, "get_tokens_batch"):
        hamer_outputs = hamer_cache.get_tokens_batch(datasets, boxes_list, valid_list)
    else:
        hamer_outputs = [
            hamer_cache.get_tokens(dataset, boxes, valid)
            for dataset, boxes, valid in zip(datasets, boxes_list, valid_list)
        ]

    prepared = []
    for dataset, detection, hamer_output in zip(datasets, detections, hamer_outputs):
        keypoints_2d, boxes, bbox_valid_2d, _ = detection
        hamer_tokens, hamer_valid = hamer_output
        bbox = torch.zeros(1, 2, window, 4)
        bbox_valid = torch.zeros(1, 2, window, dtype=torch.bool)
        bbox[0] = boxes / rectifier.output_size
        bbox_valid[0] = bbox_valid_2d
        keypoints = keypoints_2d.unsqueeze(0).float()
        keypoints[..., :2] = keypoints[..., :2] / rectifier.output_size
        images = torch.stack([
            torch.from_numpy(dataset.image(frame)).permute(2, 0, 1).float() / 255.0
            for frame in dataset.frames
        ])
        target_mano, target_camera = dataset.targets(rectifier)
        raw_extrinsics = []
        raw_intrinsics = []
        for frame in dataset.frames:
            pose = dataset.camera_pose(frame)
            raw_extrinsics.append(_se3(torch.tensor(pose["quaternion_wxyz"]), torch.tensor(pose["translation_xyz"])))
            _, _, intrinsic = dataset._calibration(frame)
            raw_intrinsics.append(torch.as_tensor(intrinsic, dtype=torch.float32))
        joints = torch.zeros(2, window, 16, 3)
        joint_valid = torch.zeros(2, window, 16, dtype=torch.bool)
        for hi, hand in enumerate(_HANDS):
            for ti, frame in enumerate(dataset.frames):
                pose = dataset.hands(frame).get(hand, {}).get("mano_pose")
                if pose is None:
                    continue
                wrist = torch.tensor(pose["wrist_xform"], dtype=torch.float32)
                components, mean = rectifier.pca_components(hand)
                local = (torch.tensor(pose["thetas"], dtype=torch.float32) @ components + mean).reshape(15, 3)
                decoded = rectifier.mano_joints(hand, wrist[:3].unsqueeze(0), wrist[3:].unsqueeze(0), local.unsqueeze(0))[0]
                joints[hi, ti] = decoded[list(_PUBLIC_FROM_MANO)]
                joint_valid[hi, ti] = True
        joints = world_points_to_first_camera(joints, raw_extrinsics[0], _FRAME_TRANSFORM)
        prepared.append(_PreparedWindow(
            hamer_tokens=hamer_tokens.unsqueeze(0), hamer_valid=hamer_valid.unsqueeze(0),
            keypoints=keypoints, bbox=bbox, bbox_valid=bbox_valid,
            target_mano=target_mano, target_camera=target_camera, images=images,
            camera_extrinsics=torch.stack(raw_extrinsics),
            camera_intrinsics=torch.stack(raw_intrinsics),
            camera_valid=torch.ones(window, dtype=torch.bool),
            hand_joints3d=joints, joint_valid=joint_valid, joint_confidence=None,
        ))
    return prepared


def sample_batch(
    *,
    clip_ids,
    window,
    hamer_cache,
    handbox_cache,
    vggt,
    rectifier,
    seeds: list[int],
    window_starts: list[int | None] | None = None,
) -> TrainSample:
    """Prepare a local training batch; extract VGGT online per full window."""
    if not seeds:
        raise ValueError("seeds must not be empty")
    starts = window_starts or [None] * len(seeds)
    if len(starts) != len(seeds):
        raise ValueError("window_starts must match seeds")
    prepared = _prepare_windows_batch(
        clip_ids=clip_ids,
        window=window,
        hamer_cache=hamer_cache,
        handbox_cache=handbox_cache,
        rectifier=rectifier,
        seeds=seeds,
        window_starts=starts,
    )
    if hasattr(vggt, "get_tokens"):
        raise RuntimeError("sample_batch requires online VGGT for batched extraction")
    images = torch.stack([item.images for item in prepared], dim=0)
    tokens = _run_vggt(vggt, images)
    return collate_samples(
        [_finish_window(item, token.cpu()) for item, token in zip(prepared, tokens)]
    )


def collate_samples(samples: list[TrainSample]) -> TrainSample:
    """Stack independent windows into one batch without mixing time axes."""
    if not samples:
        raise ValueError("samples must not be empty")
    return TrainSample(
        hamer_tokens=torch.cat([sample.hamer_tokens for sample in samples], dim=0),
        hamer_valid=torch.cat([sample.hamer_valid for sample in samples], dim=0),
        vggt_tokens=torch.cat([sample.vggt_tokens for sample in samples], dim=0),
        keypoints=torch.cat([sample.keypoints for sample in samples], dim=0),
        bbox=torch.cat([sample.bbox for sample in samples], dim=0),
        bbox_valid=torch.cat([sample.bbox_valid for sample in samples], dim=0),
        images=torch.cat([sample.images for sample in samples], dim=0),
        target_mano=ManoFlowState(
            translation=torch.cat(
                [sample.target_mano.translation for sample in samples], dim=0
            ),
            root_rotation=torch.cat(
                [sample.target_mano.root_rotation for sample in samples], dim=0
            ),
            local_rotation=torch.cat(
                [sample.target_mano.local_rotation for sample in samples], dim=0
            ),
        ),
        target_camera=CameraFlowState(
            rotation=torch.cat([sample.target_camera.rotation for sample in samples], dim=0),
            translation=torch.cat([sample.target_camera.translation for sample in samples], dim=0),
        ),
        camera_extrinsics=torch.cat([sample.camera_extrinsics for sample in samples], dim=0),
        camera_intrinsics=torch.cat([sample.camera_intrinsics for sample in samples], dim=0),
        camera_valid=torch.cat([sample.camera_valid for sample in samples], dim=0),
        hand_joints3d=torch.cat([sample.hand_joints3d for sample in samples], dim=0),
        joint_valid=torch.cat([sample.joint_valid for sample in samples], dim=0),
        joint_confidence=None,
    )
