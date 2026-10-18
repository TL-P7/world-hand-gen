"""Rotation representation conversions for flow and geometry losses.

6D rotation stores the first two columns of the rotation matrix contiguously.
``rotation_6d_to_matrix`` recovers a full SO(3) matrix via Gram-Schmidt;
``matrix_to_axis_angle`` is the SO(3) log map used to feed MANO; and
``geodesic_distance`` measures the principal angle between two rotations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """Map ``[...,6]`` continuous 6D rotation to ``[...,3,3]`` matrix."""
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Map ``[...,3,3]`` rotation matrix to ``[...,6]`` (first two columns).

    Column-major: the first three elements are column 0, the last three are
    column 1, matching ``rotation_6d_to_matrix``.
    """
    return matrix[..., :, :2].transpose(-2, -1).reshape(*matrix.shape[:-2], 6)


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Rodrigues formula: ``[...,3]`` axis-angle to ``[...,3,3]`` matrix."""
    theta = axis_angle.norm(dim=-1)  # [...]
    axis = axis_angle / theta.clamp_min(1e-8).unsqueeze(-1)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    k = torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )
    k2 = k @ k
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    theta = theta.unsqueeze(-1).unsqueeze(-1)
    return eye + theta.sin() * k + (1 - theta.cos()) * k2


def quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    """Map ``[...,4]`` wxyz quaternion to ``[...,3,3]`` rotation matrix."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """SO(3) log map with finite gradients at zero and a stable pi branch."""
    # Recover a unit quaternion by selecting the largest diagonal-based
    # component. Unlike trace/acos, this has no singular derivative at identity.
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    q_squared = torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), dim=-1).clamp_min(0)
    index = q_squared.argmax(dim=-1)
    q = torch.stack((
        torch.stack((1 + m00 + m11 + m22, matrix[..., 2, 1] - matrix[..., 1, 2], matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 1, 0] - matrix[..., 0, 1]), dim=-1),
        torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2], 1 + m00 - m11 - m22, matrix[..., 0, 1] + matrix[..., 1, 0], matrix[..., 0, 2] + matrix[..., 2, 0]), dim=-1),
        torch.stack((matrix[..., 0, 2] - matrix[..., 2, 0], matrix[..., 0, 1] + matrix[..., 1, 0], 1 - m00 + m11 - m22, matrix[..., 1, 2] + matrix[..., 2, 1]), dim=-1),
        torch.stack((matrix[..., 1, 0] - matrix[..., 0, 1], matrix[..., 0, 2] + matrix[..., 2, 0], matrix[..., 1, 2] + matrix[..., 2, 1], 1 - m00 - m11 + m22), dim=-1),
    ), dim=-2)
    q = q.gather(-2, index[..., None, None].expand(*index.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
    q = torch.where(q[..., :1] < 0, -q, q)
    half_angle = torch.atan2(q[..., 1:].norm(dim=-1), q[..., :1].squeeze(-1))
    scale = 2 / torch.sinc(half_angle / torch.pi)
    return q[..., 1:] * scale.unsqueeze(-1)


def geodesic_distance(first: Tensor, second: Tensor) -> Tensor:
    """Principal SO(3) angle in radians between two rotation matrices."""
    relative = first.transpose(-2, -1) @ second
    skew = torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), dim=-1)
    sine = 0.5 * skew.norm(dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.atan2(sine, cosine)
