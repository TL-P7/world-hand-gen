"""Shared camera-frame transforms used by dataset adaptors.

All matrices act on column vectors.  A raw dataset pose ``E_t`` has the
semantics ``x_world = E_t @ x_camera``.  The canonical training target is the
current-camera-to-first-camera transform in the dataset's model camera axes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.rotation import matrix_to_rotation_6d
from src.model.flow.state import CameraFlowState

CAMERA_EXTRINSICS_SEMANTICS = "world_from_camera"
TARGET_CAMERA_SEMANTICS = "current_model_camera_to_first_model_camera"
TARGET_CAMERA_EQUATION = "U_t = F @ inverse(E_0) @ E_t @ inverse(F)"
TARGET_CAMERA_EQUATION_NO_FRAME = "U_t = inverse(E_0) @ E_t"
TARGET_CAMERA_REFERENCE = "window_first_model_camera"


def relative_camera_transforms(
    world_from_camera: Tensor,
    frame_transform: Tensor | None = None,
) -> Tensor:
    """Return canonical current-camera-to-first-camera transforms.

    ``world_from_camera`` has shape ``[T,4,4]`` or ``[B,T,4,4]``.  ``F`` is an
    optional fixed change of camera axes from the dataset camera to the model
    camera.  The returned matrices satisfy ``x_first = U_t @ x_current``.
    """
    if world_from_camera.ndim not in (3, 4) or world_from_camera.shape[-2:] != (4, 4):
        raise ValueError("world_from_camera must have shape [T,4,4] or [B,T,4,4]")
    first_inverse = torch.linalg.inv(world_from_camera[..., :1, :, :])
    relative = first_inverse @ world_from_camera
    if frame_transform is not None:
        if frame_transform.shape != (4, 4):
            raise ValueError("frame_transform must have shape [4,4]")
        frame_transform = frame_transform.to(
            device=world_from_camera.device, dtype=world_from_camera.dtype
        )
        relative = frame_transform @ relative @ torch.linalg.inv(frame_transform)
    return relative


def camera_flow_from_extrinsics(
    world_from_camera: Tensor,
    frame_transform: Tensor | None = None,
) -> CameraFlowState:
    """Encode canonical relative camera transforms as a flow state."""
    relative = relative_camera_transforms(world_from_camera, frame_transform)
    return CameraFlowState(
        rotation=matrix_to_rotation_6d(relative[..., :3, :3]),
        translation=relative[..., :3, 3],
    )


def world_points_to_first_camera(
    points_world: Tensor,
    first_world_from_camera: Tensor,
    frame_transform: Tensor | None = None,
) -> Tensor:
    """Transform world-space points into the first model-camera coordinate frame."""
    if first_world_from_camera.shape != (4, 4):
        raise ValueError("first_world_from_camera must have shape [4,4]")
    model_from_world = torch.linalg.inv(first_world_from_camera)
    if frame_transform is not None:
        if frame_transform.shape != (4, 4):
            raise ValueError("frame_transform must have shape [4,4]")
        frame_transform = frame_transform.to(
            device=points_world.device, dtype=points_world.dtype
        )
        model_from_world = frame_transform @ model_from_world.to(
            device=points_world.device, dtype=points_world.dtype
        )
    else:
        model_from_world = model_from_world.to(
            device=points_world.device, dtype=points_world.dtype
        )
    rotation = model_from_world[:3, :3]
    translation = model_from_world[:3, 3]
    return torch.einsum("ij,...j->...i", rotation, points_world) + translation
