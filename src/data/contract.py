"""Dataset-neutral window and supervision contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

from src.model.flow.state import CameraFlowState, ManoFlowState

# Public project order: wrist, then three joints for thumb/index/middle/ring/little.
JOINT_ORDER = (
    "wrist",
    "thumb_1", "thumb_2", "thumb_3",
    "index_1", "index_2", "index_3",
    "middle_1", "middle_2", "middle_3",
    "ring_1", "ring_2", "ring_3",
    "little_1", "little_2", "little_3",
)


@dataclass
class WindowSample:
    """One sampled window with optional conditions and supervision targets.

    Tensors use a leading batch dimension. ``target_joints3d`` and
    ``joint_valid`` are in :data:`JOINT_ORDER`; adapters own their coordinate
    frame and unit contract and must not silently convert them here.
    """

    hamer_tokens: Tensor
    hamer_valid: Tensor
    vggt_tokens: Tensor
    vggt_valid: Tensor
    keypoints: Tensor
    bbox: Tensor
    bbox_valid: Tensor
    images: Tensor | None = None  # [B,T,3,H,W], adapter image convention
    camera_extrinsics: Tensor | None = None  # [B,T,4,4], raw world_from_camera
    camera_intrinsics: Tensor | None = None  # [B,T,3,3], image pixel coordinates
    camera_valid: Tensor | None = None  # [B,T]
    hand_joints3d: Tensor | None = None  # [B,2,T,16,3], public order
    joint_valid: Tensor | None = None  # [B,2,T,16]
    joint_confidence: Tensor | None = None  # [B,2,T,16], None when unavailable
    target_mano: ManoFlowState | None = None
    # Relative transform U_t: x_first_model_camera = U_t @ x_current_model_camera.
    target_camera: CameraFlowState | None = None
    # Backward-compatible alias for the canonical hand_joints3d field.
    target_joints3d: Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.hand_joints3d is None and self.target_joints3d is not None:
            self.hand_joints3d = self.target_joints3d
        if self.target_joints3d is None and self.hand_joints3d is not None:
            self.target_joints3d = self.hand_joints3d


class DatasetAdapter(Protocol):
    """Dataset-owned sampling interface consumed by training."""

    def sample(self, *, seed: int, batch_size: int, window: int) -> WindowSample:
        ...

    def describe(self) -> dict[str, Any]:
        ...
