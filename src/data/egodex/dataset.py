"""Lazy access to one consecutive EgoDex window.

``joints()`` returns ``(xyz, valid)`` with shapes ``[2,T,16,3]`` and
``[2,T,16]`` (left, right). The joint order matches fit_egodex's OP_INDICES:
wrist, thumb, index, middle, ring, little; three non-tip joints per finger.
This is not the native 16-joint order returned by the project's ManoWrapper.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np

from src.config import EGODEX_ROOT

JOINT_SUFFIXES = (
    "Hand",
    "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip",
    "IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip",
    "MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip",
    "RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip",
    "LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip",
)


class EgoDexDataset:
    """Read only RGB frames, raw world-space OP-16 joints, and validity.

    Missing joints remain NaN and invalid. Available confidence must be finite
    and positive; absent confidence falls back to finite joint coordinates.
    No confidence weighting or coordinate/unit conversion is performed.
    """

    def __init__(self, episode_id: str, frames=None, root: Path | None = None):
        import h5py

        root = EGODEX_ROOT if root is None else Path(root)
        episode = Path(episode_id)
        if episode.is_absolute() or ".." in episode.parts or len(episode.parts) != 3:
            raise ValueError("episode_id must be 'part/task/episode' relative to the root")
        self.episode_id = str(episode)
        self.annotation_path = root / episode.parent / f"{episode.name}.hdf5"
        self.video_path = root / episode.parent / f"{episode.name}.mp4"
        for path in (self.annotation_path, self.video_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with h5py.File(self.annotation_path, "r") as annotation:
            lengths = set()
            for side in ("left", "right"):
                for suffix in JOINT_SUFFIXES:
                    name = f"transforms/{side}{suffix}"
                    if name not in annotation:
                        continue
                    shape = annotation[name].shape
                    if len(shape) != 3 or shape[1:] != (4, 4):
                        raise ValueError(f"invalid EgoDex transform shape: {name} {shape}")
                    lengths.add(shape[0])
            if len(lengths) != 1:
                raise ValueError("EgoDex hand transforms must have one common frame count")
            self.num_frames = lengths.pop()
        self.frames = list(range(self.num_frames)) if frames is None else list(frames)
        if not self.frames or any(
            not isinstance(frame, Integral) or isinstance(frame, bool)
            or frame < 0 or frame >= self.num_frames for frame in self.frames
        ):
            raise ValueError("frames must select nonempty in-range integer frame indices")
        if self.frames != list(range(self.frames[0], self.frames[0] + len(self.frames))):
            raise ValueError("frames must select a consecutive window")
        self._images = {}
        self._joints = None
        self._joint_confidence = None
        self._camera = None

    def image(self, frame: int) -> np.ndarray:
        """Return an unrotated uint8 RGB frame in its original resolution."""
        import cv2

        if frame not in self.frames:
            raise ValueError("frame is outside the selected window")
        if frame not in self._images:
            capture = cv2.VideoCapture(str(self.video_path))
            try:
                if not capture.isOpened():
                    raise OSError(f"cannot open EgoDex video: {self.video_path}")
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if count != self.num_frames:
                    raise ValueError(
                        f"EgoDex video/annotation frame mismatch: {count} != {self.num_frames}"
                    )
                if frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame)):
                    raise OSError(f"cannot seek EgoDex video to frame {frame}")
                ok, bgr = capture.read()
                if not ok:
                    raise OSError(f"cannot decode EgoDex frame {frame}: {self.video_path}")
                self._images[frame] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            finally:
                capture.release()
        return self._images[frame]

    def joints(self) -> tuple[np.ndarray, np.ndarray]:
        """Read OP-16 translation columns and validity for the selected window."""
        self.joints_with_confidence()
        return self._joints

    def joints_with_confidence(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Read OP-16 points, validity, and optional source confidence."""
        import h5py

        if self._joints is None:
            xyz = np.full((2, len(self.frames), 16, 3), np.nan, dtype=np.float32)
            valid = np.zeros(xyz.shape[:-1], dtype=bool)
            confidence_values = np.full(xyz.shape[:-1], np.nan, dtype=np.float32)
            have_confidence = False
            window = slice(self.frames[0], self.frames[-1] + 1)
            with h5py.File(self.annotation_path, "r") as annotation:
                for hand_index, side in enumerate(("left", "right")):
                    for joint_index, suffix in enumerate(JOINT_SUFFIXES):
                        name = f"{side}{suffix}"
                        transform = f"transforms/{name}"
                        if transform not in annotation:
                            continue
                        points = annotation[transform][window, :3, 3]
                        xyz[hand_index, :, joint_index] = points
                        mask = np.isfinite(points).all(axis=-1)
                        confidence = f"confidences/{name}"
                        if confidence in annotation:
                            if annotation[confidence].shape != (self.num_frames,):
                                raise ValueError(f"invalid EgoDex confidence shape: {confidence}")
                            values = annotation[confidence][window]
                            confidence_values[hand_index, :, joint_index] = values
                            have_confidence = True
                            mask &= np.isfinite(values) & (values > 0)
                        valid[hand_index, :, joint_index] = mask
            self._joints = (xyz, valid)
            self._joint_confidence = confidence_values if have_confidence else None
        return (*self._joints, self._joint_confidence)

    def camera(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Read raw ``world_from_camera`` matrices and pixel intrinsics."""
        import h5py

        if self._camera is None:
            window = slice(self.frames[0], self.frames[-1] + 1)
            with h5py.File(self.annotation_path, "r") as annotation:
                transforms = annotation["transforms/camera"][window]
                if transforms.shape != (len(self.frames), 4, 4):
                    raise ValueError("invalid EgoDex camera transform shape")
                intrinsic = annotation["camera/intrinsic"][:]
            fallback = intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            if fallback:
                intrinsic = np.array(
                    [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            self._camera = (
                np.asarray(transforms, dtype=np.float32),
                np.broadcast_to(np.asarray(intrinsic, dtype=np.float32), (len(self.frames), 3, 3)).copy(),
                fallback,
            )
        return self._camera
