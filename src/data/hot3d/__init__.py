"""Raw HOT3D Aria loading and per-frame rectification.

The raw loader returns untransformed FISHEYE624 frames. ``Hot3DRectifier``
holds the rectification primitives (FISHEYE624 -> linear pinhole, clockwise
90-degree rotation, box re-projection) plus lazily-loaded MANO decoding. It no
longer performs whole-window decoding; per-frame access is provided by
``Hot3DDataset`` instead.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import DATA_ROOT, MANO_ROOT

RGB_CAMERA_LABEL = "214-1"
DEFAULT_ROOT = DATA_ROOT


@dataclass
class Hot3DFrame:
    """A raw frame: untransformed RGB image plus hands/cameras/info JSON."""

    clip_id: str
    frame: int
    image: np.ndarray
    hands: dict
    cameras: dict
    info: dict


def clip_ids(root: Path | None = None) -> list[str]:
    root = DATA_ROOT if root is None else Path(root)
    return sorted(p.name[len("clip-"):] for p in (root / "train_aria").glob("clip-*"))


def frame_ids(clip_id: str, root: Path | None = None) -> list[int]:
    root = DATA_ROOT if root is None else Path(root)
    directory = root / "train_aria" / f"clip-{clip_id}"
    return sorted(
        int(p.name.split(".")[0])
        for p in directory.glob(f"*.image_{RGB_CAMERA_LABEL}.jpg")
    )


def load_frame(clip_id: str, frame: int, root: Path | None = None) -> Hot3DFrame:
    root = DATA_ROOT if root is None else Path(root)
    directory = root / "train_aria" / f"clip-{clip_id}"
    with Image.open(directory / f"{frame:06d}.image_{RGB_CAMERA_LABEL}.jpg") as image:
        rgb = np.asarray(image.convert("RGB"))
    return Hot3DFrame(
        clip_id=clip_id,
        frame=frame,
        image=rgb,
        hands=json.loads((directory / f"{frame:06d}.hands.json").read_text()),
        cameras=json.loads((directory / f"{frame:06d}.cameras.json").read_text()),
        info=json.loads((directory / f"{frame:06d}.info.json").read_text()),
    )


class Hot3DRectifier:
    """Rectification and MANO decoding primitives for HOT3D Aria frames."""

    def __init__(self, *, output_size: int = 384, camera_label: str = RGB_CAMERA_LABEL):
        self.output_size = output_size
        self.camera_label = camera_label
        self._layers = None
        self._decoders = None
        self._upright_rotation = None

    def make_calibration(self, calibration_json):
        """Build FISHEYE624 source, linear destination, and rotated K."""
        from projectaria_tools.core import calibration, sophus

        source = calibration.CameraCalibration(
            str(calibration_json["label"]),
            calibration.CameraModelType.FISHEYE624,
            np.asarray(calibration_json["projection_params"], dtype=np.float64),
            sophus.SE3(),
            int(calibration_json["image_width"]),
            int(calibration_json["image_height"]),
            None,
            float(calibration_json["max_solid_angle"]),
            str(calibration_json["serial_number"]),
        )
        focal = (
            float(calibration_json["projection_params"][0])
            * self.output_size
            / int(calibration_json["image_width"])
        )
        destination = calibration.get_linear_camera_calibration(
            self.output_size, self.output_size, focal, "hot3d-linear"
        )
        intrinsics = np.array(
            [
                [focal, 0.0, (self.output_size - 1) / 2],
                [0.0, focal, (self.output_size - 1) / 2],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = np.zeros((3, 3))
        rotated[0, 0] = intrinsics[1, 1]
        rotated[1, 1] = intrinsics[0, 0]
        rotated[0, 2] = self.output_size - 1 - intrinsics[1, 2]
        rotated[1, 2] = intrinsics[0, 2]
        rotated[2, 2] = 1.0
        return source, destination, rotated

    def rectify_image(self, raw, source, destination):
        """Rectify a raw RGB frame to upright linear pixels."""
        from projectaria_tools.core import calibration as aria_calibration, sophus

        if self._upright_rotation is None:
            # The dataset convention is a clockwise 90-degree image rotation.
            self._upright_rotation = sophus.SO3.from_matrix(
                np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            )
        if hasattr(aria_calibration, "distort_by_calibration_and_apply_rotation"):
            rectified = aria_calibration.distort_by_calibration_and_apply_rotation(
                raw, destination, source, self._upright_rotation
            )
            return np.ascontiguousarray(rectified)
        rectified = aria_calibration.distort_by_calibration(raw, destination, source)
        return np.ascontiguousarray(np.rot90(rectified, k=3))

    def rectify_box(self, box, source, destination):
        """Return visible upright xyxy pixel bounds, or None outside the image.

        Warp the box's pixel support with the same calibration and CW90 as RGB.
        Nearest-neighbor label resampling avoids extrapolating fisheye rays
        outside the source image. Bounds are inclusive pixel centers.
        """
        from projectaria_tools.core import calibration

        box = np.asarray(box, dtype=np.float64)
        if not np.isfinite(box).all():
            return None
        width, height = source.get_image_size()
        x1, y1 = np.ceil(np.maximum(box[:2], 0)).astype(int)
        x2, y2 = np.floor(np.minimum(box[2:], [width - 1, height - 1])).astype(int)
        if x2 <= x1 or y2 <= y1:
            return None
        support = np.zeros((height, width), dtype=np.uint8)
        support[y1:y2 + 1, x1:x2 + 1] = 1
        upright = np.rot90(
            calibration.distort_label_by_calibration(support, destination, source), k=3
        )
        ys, xs = np.nonzero(upright)
        if xs.size == 0 or xs.max() == xs.min() or ys.max() == ys.min():
            return None
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    def _ensure_mano(self) -> None:
        if self._layers is not None:
            return
        import smplx

        self._layers = {
            "left": smplx.create(
                str(MANO_ROOT), model_type="mano", is_rhand=False, use_pca=False,
                flat_hand_mean=True
            ).eval(),
            "right": smplx.create(
                str(MANO_ROOT), model_type="mano", is_rhand=True, use_pca=False,
                flat_hand_mean=True
            ).eval(),
        }
        self._decoders = {}
        for hand in ("left", "right"):
            with open(MANO_ROOT / "mano" / f"MANO_{hand.upper()}.pkl", "rb") as handle:
                data = pickle.load(handle, encoding="latin1")
            self._decoders[hand] = (
                torch.as_tensor(data["hands_components"][:15], dtype=torch.float32),
                torch.as_tensor(data["hands_mean"], dtype=torch.float32),
            )

    def pca_components(self, hand):
        """Return the MANO PCA basis ``(components [15,45], mean [45])``."""
        self._ensure_mano()
        return self._decoders[hand]

    def mano_joints(self, hand, root, transl, local):
        """Forward MANO for a single hand; ``root`` is axis-angle, ``local`` is
        ``[...,15,3]`` axis-angle, ``transl`` is ``[...,3]``."""
        self._ensure_mano()
        layer = self._layers[hand]
        with torch.no_grad():
            output = layer(
                global_orient=root,
                hand_pose=local.reshape(root.shape[0], -1),
                betas=torch.zeros(root.shape[0], 10, dtype=torch.float32),
                transl=transl,
            )
        return output.joints
