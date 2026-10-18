"""Lazy HOT3D window dataset.

Reads raw frames and rectifies them on demand, caching per-frame results so a
frame is never read or rectified twice within a window. This is the single
place that reads HOT3D from disk and applies the upright transform.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None
else:
    cv2.setNumThreads(1)

from src.data.camera import camera_flow_from_extrinsics
from src.geometry.rotation import axis_angle_to_matrix, matrix_to_rotation_6d, quaternion_wxyz_to_matrix
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.config import DATA_ROOT
from . import RGB_CAMERA_LABEL


class Hot3DDataset:
    """Lazy per-frame access to rectified images and hand boxes."""

    def __init__(
        self, clip_id, frames, rectifier, root: Path | None = None,
        image_cache_root: Path | None = None,
    ):
        root = DATA_ROOT if root is None else Path(root)
        self.clip_id = clip_id
        self.frames = list(frames)
        self.rectifier = rectifier
        self.directory = Path(root) / "train_aria" / f"clip-{clip_id}"
        self.image_cache_root = None
        self._cache_keys = {}
        if image_cache_root is not None:
            cache_root = Path(image_cache_root).resolve()
            for protected in (Path(root).resolve(), DATA_ROOT.parent.resolve()):
                if cache_root == protected or protected in cache_root.parents:
                    raise ValueError("preprocessed cache must be outside the data directory")
            signature = hashlib.sha256()
            signature.update(str(Path(root).resolve()).encode())
            signature.update(Path(__file__).read_bytes())
            signature.update(Path(__file__).with_name("__init__.py").read_bytes())
            signature.update(str(cv2.__version__ if cv2 is not None else getattr(Image, "__version__", "unknown")).encode())
            self.image_cache_root = cache_root / signature.hexdigest()[:16]
        self._images = {}
        self._calibrations = {}
        self._boxes = {}
        self._hands = {}
        self._camera_poses = {}
        self._raw_cameras = {}

    def _cameras(self, frame):
        if frame not in self._raw_cameras:
            self._raw_cameras[frame] = json.loads(
                (self.directory / f"{frame:06d}.cameras.json").read_text()
            )
        return self._raw_cameras[frame]

    def _raw_hands(self, frame):
        if frame not in self._hands:
            self._hands[frame] = json.loads(
                (self.directory / f"{frame:06d}.hands.json").read_text()
            )
        return self._hands[frame]

    def _calibration(self, frame):
        if frame not in self._calibrations:
            cameras = self._cameras(frame)
            self._calibrations[frame] = self.rectifier.make_calibration(
                cameras[RGB_CAMERA_LABEL]["calibration"]
            )
        return self._calibrations[frame]

    def _image_cache_path(self, frame) -> Path | None:
        if self.image_cache_root is None:
            return None
        if frame not in self._cache_keys:
            sources = [
                self.directory / f"{frame:06d}.{suffix}"
                for suffix in (f"image_{RGB_CAMERA_LABEL}.jpg", "cameras.json", "hands.json")
            ]
            stats = [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources]
            self._cache_keys[frame] = hashlib.sha256(repr(stats).encode()).hexdigest()[:16]
        return (
            self.image_cache_root / str(self.rectifier.output_size) / self.clip_id
            / f"{frame:06d}-{self._cache_keys[frame]}.npy"
        )

    def _box_cache_path(self, hand, frame) -> Path | None:
        image_path = self._image_cache_path(frame)
        if image_path is None:
            return None
        return image_path.with_name(f"{image_path.stem}_{hand}_box.npy")

    def _save_cached_array(self, path: Path, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.save(handle, value, allow_pickle=False)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def image(self, frame) -> np.ndarray:
        """Return the upright rectified RGB image for one frame."""
        if frame not in self._images:
            cache_path = self._image_cache_path(frame)
            if cache_path is not None and cache_path.is_file():
                cached = np.load(cache_path, allow_pickle=False)
                expected = (self.rectifier.output_size, self.rectifier.output_size, 3)
                if cached.shape != expected or cached.dtype != np.uint8:
                    raise ValueError(f"invalid rectified image cache: {cache_path}")
                self._images[frame] = np.ascontiguousarray(cached)
                return self._images[frame]
            source, destination, _ = self._calibration(frame)
            path = self.directory / f"{frame:06d}.image_{RGB_CAMERA_LABEL}.jpg"
            if cv2 is None:
                with Image.open(path) as image:
                    raw = np.asarray(image.convert("RGB"))
            else:
                raw_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if raw_bgr is None:
                    raise OSError(f"failed to decode image: {path}")
                raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            rectified = self.rectifier.rectify_image(raw, source, destination)
            if cache_path is not None:
                self._save_cached_array(cache_path, rectified)
            self._images[frame] = rectified
        return self._images[frame]

    def hand_box(self, hand, frame):
        """Return the upright rectified ``xyxy`` box for one hand, or ``None``."""
        key = (hand, frame)
        if key not in self._boxes:
            cache_path = self._box_cache_path(hand, frame)
            if cache_path is not None and cache_path.is_file():
                value = np.load(cache_path, allow_pickle=False)
                if value.shape not in ((0,), (4,)) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"invalid rectified box cache: {cache_path}")
                self._boxes[key] = value if value.size else None
                return self._boxes[key]
            hands = self._raw_hands(frame)
            raw_box = hands.get(hand, {}).get("boxes_amodal", {}).get(RGB_CAMERA_LABEL)
            if raw_box is None:
                self._boxes[key] = None
            else:
                source, destination, _ = self._calibration(frame)
                self._boxes[key] = self.rectifier.rectify_box(raw_box, source, destination)
            if cache_path is not None:
                value = np.empty(0, dtype=np.float32) if self._boxes[key] is None else self._boxes[key]
                self._save_cached_array(cache_path, value)
        return self._boxes[key]

    def hands(self, frame) -> dict:
        """Return the raw per-frame hands annotation."""
        return self._raw_hands(frame)

    def camera_pose(self, frame) -> dict:
        """Return the raw camera-to-world pose for the RGB camera."""
        if frame not in self._camera_poses:
            cameras = self._cameras(frame)
            self._camera_poses[frame] = cameras[RGB_CAMERA_LABEL]["T_world_from_camera"]
        return self._camera_poses[frame]

    def targets(self, rectifier):
        """Build first-model-camera camera and MANO flow targets.

        Raw poses are ``E_t = world_from_camera``.  The camera target is
        ``U_t = F @ inverse(E_0) @ E_t @ inverse(F)`` and acts as
        ``x_first_model_camera = U_t @ x_current_model_camera``.
        """
        frame_transform = torch.eye(4)
        frame_transform[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])

        def se3(pose):
            out = torch.eye(4)
            out[:3, :3] = quaternion_wxyz_to_matrix(torch.tensor(pose["quaternion_wxyz"], dtype=torch.float32))
            out[:3, 3] = torch.tensor(pose["translation_xyz"], dtype=torch.float32)
            return out

        poses = torch.stack([se3(self.camera_pose(frame)) for frame in self.frames])
        camera_target = camera_flow_from_extrinsics(poses, frame_transform)

        translation = torch.zeros(2, len(self.frames), 3)
        root_rotation = torch.zeros(2, len(self.frames), 6)
        local_rotation = torch.zeros(2, len(self.frames), 15, 6)
        world_to_upright = frame_transform @ torch.linalg.inv(poses[0])
        for hi, hand in enumerate(("left", "right")):
            components, mean = rectifier.pca_components(hand)
            rest_wrist = rectifier.mano_joints(
                hand, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 15, 3)
            )[0, 0]
            valid_indices = []
            roots = []
            translations = []
            local_axes = []
            for ti, frame in enumerate(self.frames):
                pose = self.hands(frame).get(hand, {}).get("mano_pose")
                if pose is None:
                    continue
                valid_indices.append(ti)
                theta = torch.tensor(pose["thetas"], dtype=torch.float32)
                wrist = torch.tensor(pose["wrist_xform"], dtype=torch.float32)
                local_axis = (theta @ components + mean).reshape(15, 3)
                local_rotation[hi, ti] = matrix_to_rotation_6d(axis_angle_to_matrix(local_axis))
                roots.append(wrist[:3])
                translations.append(wrist[3:])
                local_axes.append(local_axis)
                root_world = axis_angle_to_matrix(wrist[:3])
                root_upright = world_to_upright[:3, :3] @ root_world
                root_rotation[hi, ti] = matrix_to_rotation_6d(root_upright)
            if valid_indices:
                roots = torch.stack(roots)
                translations = torch.stack(translations)
                local_axes = torch.stack(local_axes)
                source = rectifier.mano_joints(hand, roots, translations, local_axes)
                desired = torch.einsum(
                    "ij,nj->ni", world_to_upright[:3, :3], source[:, 0]
                ) + world_to_upright[:3, 3]
                for row, ti in enumerate(valid_indices):
                    translation[hi, ti] = desired[row] - rest_wrist
        return (
            ManoFlowState(translation.unsqueeze(0), root_rotation.unsqueeze(0), local_rotation.unsqueeze(0)),
            camera_target.map(lambda value: value.unsqueeze(0)),
        )
