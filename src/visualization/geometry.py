"""Pure geometry helpers for visualizing generated MANO sequences.

The functions in this module intentionally accept already-decoded tensors only.
They do not read HOT3D files, instantiate MANO, or load a camera calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.geometry.rotation import rotation_6d_to_matrix


def _tensor(value: Any, name: str) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach()
    elif isinstance(value, np.ndarray):
        result = torch.from_numpy(value)
    else:
        result = torch.as_tensor(value)
    if not result.is_floating_point():
        result = result.float()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _batched_vertices(vertices: Any) -> Tensor:
    result = _tensor(vertices, "vertices")
    if result.ndim == 4:
        result = result.unsqueeze(0)
    if result.ndim != 5 or result.shape[-1] != 3:
        raise ValueError("vertices must have shape [B,2,T,V,3] or [2,T,V,3]")
    if result.shape[1] != 2:
        raise ValueError("vertices must contain exactly two hand streams")
    return result


def _batched_camera(rotation: Any, translation: Any) -> tuple[Tensor, Tensor]:
    camera_rotation = _tensor(rotation, "camera_rotation")
    camera_translation = _tensor(translation, "camera_translation")
    if camera_rotation.ndim == 2:
        camera_rotation = camera_rotation.unsqueeze(0)
    if camera_translation.ndim == 2:
        camera_translation = camera_translation.unsqueeze(0)
    if camera_rotation.ndim != 3 or camera_rotation.shape[-1] != 6:
        raise ValueError("camera_rotation must have shape [B,T,6] or [T,6]")
    if camera_translation.ndim != 3 or camera_translation.shape[-1] != 3:
        raise ValueError("camera_translation must have shape [B,T,3] or [T,3]")
    if camera_rotation.shape[:2] != camera_translation.shape[:2]:
        raise ValueError("camera rotation and translation batch/time dimensions must match")
    return camera_rotation, camera_translation


def align_scene_scale(
    prediction_joints: Any,
    target_joints: Any,
    prediction_camera_translation: Any,
    target_camera_translation: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit one positive scene scale without translation or rotation alignment.

    The least-squares scalar is fitted jointly to all hand joints and camera
    translations in the supplied reference frame. This keeps hand/camera
    geometry coupled while changing only the predicted metric scale.
    """
    prediction_joints = _tensor(prediction_joints, "prediction_joints")
    target_joints = _tensor(target_joints, "target_joints")
    prediction_camera_translation = _tensor(
        prediction_camera_translation, "prediction_camera_translation"
    )
    target_camera_translation = _tensor(target_camera_translation, "target_camera_translation")
    if prediction_joints.ndim != 5 or prediction_joints.shape[-1] != 3:
        raise ValueError("joint tensors must have shape [B,2,T,J,3]")
    if target_joints.shape != prediction_joints.shape:
        raise ValueError("prediction_joints and target_joints must have identical shapes")
    if prediction_camera_translation.ndim != 3 or prediction_camera_translation.shape[-1] != 3:
        raise ValueError("camera tensors must have shape [B,T,3]")
    if target_camera_translation.shape != prediction_camera_translation.shape:
        raise ValueError("prediction and target camera tensors must have identical shapes")
    if (prediction_joints.shape[0] != prediction_camera_translation.shape[0]
            or prediction_joints.shape[2] != prediction_camera_translation.shape[1]):
        raise ValueError("joint and camera batch/time dimensions must match")
    target_joints = target_joints.to(device=prediction_joints.device, dtype=prediction_joints.dtype)
    prediction_camera_translation = prediction_camera_translation.to(
        device=prediction_joints.device, dtype=prediction_joints.dtype
    )
    target_camera_translation = target_camera_translation.to(
        device=prediction_joints.device, dtype=prediction_joints.dtype
    )
    prediction = torch.cat((
        prediction_joints.reshape(prediction_joints.shape[0], -1, 3),
        prediction_camera_translation.reshape(prediction_camera_translation.shape[0], -1, 3),
    ), dim=1)
    target = torch.cat((
        target_joints.reshape(target_joints.shape[0], -1, 3),
        target_camera_translation.reshape(target_camera_translation.shape[0], -1, 3),
    ), dim=1)
    numerator = (prediction * target).sum(dim=(1, 2))
    denominator = prediction.square().sum(dim=(1, 2))
    eps = torch.finfo(prediction.dtype).eps
    scale = (numerator / denominator.clamp_min(eps)).clamp_min(eps)
    return (
        prediction_joints * scale.view(-1, 1, 1, 1, 1),
        prediction_camera_translation * scale.view(-1, 1, 1),
        scale,
    )


def transform_vertices_to_camera(
    vertices: Any,
    camera_rotation: Any,
    camera_translation: Any,
) -> Tensor:
    """Transform reference-frame vertices into each generated camera frame.

    The flow model stores ``U_t`` as a transform from frame ``t`` model-camera
    coordinates into the first model-camera reference frame, i.e. ``x_ref =
    R_t @ x_cam + t_t``.  The returned tensor therefore applies the inverse
    transform ``x_cam = R_t.T @ (x_ref - t_t)``.

    Args:
        vertices: ``[B,2,T,V,3]`` or unbatched ``[2,T,V,3]`` reference-frame
            vertices in metres.
        camera_rotation: generated 6D rotations ``[B,T,6]`` or ``[T,6]``.
        camera_translation: generated translations ``[B,T,3]`` or ``[T,3]``.

    Returns:
        A detached float tensor with shape ``[B,2,T,V,3]``.
    """
    vertices_tensor = _batched_vertices(vertices)
    camera_rotation_tensor, camera_translation_tensor = _batched_camera(
        camera_rotation, camera_translation
    )
    if vertices_tensor.shape[0] != camera_rotation_tensor.shape[0]:
        raise ValueError("vertices and camera state batch dimensions must match")
    if vertices_tensor.shape[2] != camera_rotation_tensor.shape[1]:
        raise ValueError("vertices and camera state time dimensions must match")
    camera_rotation_tensor = camera_rotation_tensor.to(
        device=vertices_tensor.device, dtype=vertices_tensor.dtype
    )
    camera_translation_tensor = camera_translation_tensor.to(
        device=vertices_tensor.device, dtype=vertices_tensor.dtype
    )
    rotation = rotation_6d_to_matrix(camera_rotation_tensor)
    centered = vertices_tensor - camera_translation_tensor[:, None, :, None, :]
    result = torch.einsum("btji,bhtvj->bhtvi", rotation, centered)
    if not torch.isfinite(result).all():
        raise ValueError("transformed vertices contain non-finite values")
    return result


def _batched_intrinsics(intrinsics: Any, batch: int, time: int, *, device, dtype) -> Tensor:
    matrix = _tensor(intrinsics, "intrinsics").to(device=device, dtype=dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    elif matrix.ndim == 3:
        if matrix.shape[0] == time:
            matrix = matrix.unsqueeze(0)
        elif matrix.shape[0] == batch:
            matrix = matrix.unsqueeze(1)
        else:
            raise ValueError("intrinsics [N,3,3] must use N equal to batch or time")
    if matrix.ndim != 4 or matrix.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [3,3], [T,3,3], or [B,T,3,3]")
    if matrix.shape[0] not in (1, batch) or matrix.shape[1] not in (1, time):
        raise ValueError("intrinsics batch/time dimensions do not match points")
    return matrix.expand(batch, time, 3, 3)


def project_points(
    points_camera: Any,
    intrinsics: Any,
    *,
    min_depth: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project camera-frame points using supplied pinhole intrinsics.

    Returns ``(uv, depth, valid)``. ``uv`` has shape ``[B,2,T,V,2]`` and is
    finite even for points behind the camera; ``valid`` marks finite points
    whose depth is greater than ``min_depth``.
    """
    points = _batched_vertices(points_camera)
    if min_depth <= 0:
        raise ValueError("min_depth must be positive")
    matrix = _batched_intrinsics(
        intrinsics, points.shape[0], points.shape[2], device=points.device, dtype=points.dtype
    )
    pixels = torch.einsum("btij,bhtvj->bhtvi", matrix, points)
    depth = pixels[..., 2]
    valid = torch.isfinite(depth) & (depth > min_depth) & torch.isfinite(points).all(dim=-1)
    safe_depth = depth.clamp_min(min_depth)
    uv = pixels[..., :2] / safe_depth.unsqueeze(-1)
    uv = torch.where(torch.isfinite(uv), uv, torch.zeros_like(uv))
    return uv, depth, valid
