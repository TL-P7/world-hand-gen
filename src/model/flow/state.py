"""Flow-matching state dataclasses.

Both generation targets are represented in a continuous flow space:

- MANO: ``[B,2,T,3]`` translation plus ``[B,2,T,6]`` root rotation and
  ``[B,2,T,15,6]`` local rotations, all in the 6D continuous rotation form.
- Camera: ``[B,T,6]`` rotation (6D) plus ``[B,T,3]`` translation.

The 6D rotation form is intentionally not converted back to matrices here;
that conversion lives outside the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

from torch import Tensor

T = TypeVar("T", bound="ManoFlowState")


@dataclass
class ManoFlowState:
    """Two-hand MANO state in continuous flow space."""

    translation: Tensor  # [B,2,T,3]
    root_rotation: Tensor  # [B,2,T,6]
    local_rotation: Tensor  # [B,2,T,15,6]

    def map(self: T, function: Callable[[Tensor], Tensor]) -> T:
        return type(self)(
            translation=function(self.translation),
            root_rotation=function(self.root_rotation),
            local_rotation=function(self.local_rotation),
        )


@dataclass
class CameraFlowState:
    """Relative first-model-camera transform in continuous flow space.

    The decoded transform is ``U_t`` where
    ``x_first_model_camera = U_t @ x_current_model_camera``. Translation uses
    the adaptor's declared camera/world unit; rotation is represented by 6D.
    """

    rotation: Tensor  # [B,T,6]
    translation: Tensor  # [B,T,3]

    def map(self, function: Callable[[Tensor], Tensor]) -> "CameraFlowState":
        return CameraFlowState(
            rotation=function(self.rotation),
            translation=function(self.translation),
        )


@dataclass
class ConditionTokens:
    """Projected condition tokens for cross-attention key/value.

    ``hamer``, ``bbox`` and ``keypoints`` are flattened to ``[B,2T,D]``;
    ``vggt`` is ``[B,17T,D]``. Invalid positions are filled with a
    learnable missing token during encoding, so attention does not need masks.
    """

    hamer: Tensor  # [B,2T,D]
    vggt: Tensor  # [B,17T,D]
    bbox: Tensor  # [B,2T,D]
    keypoints: Tensor  # [B,2T,D]
