from __future__ import annotations

import math

import torch

from src.geometry.rotation import geodesic_distance, matrix_to_axis_angle, rotation_6d_to_matrix


def _rot_z(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identity_6d_maps_to_identity_matrix() -> None:
    matrix = rotation_6d_to_matrix(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)


def test_rotation_6d_to_matrix_is_so3() -> None:
    r6d = torch.randn(5, 6)
    matrix = rotation_6d_to_matrix(r6d)
    assert matrix.shape == (5, 3, 3)
    # orthonormal and determinant +1
    assert torch.allclose(matrix.transpose(-2, -1) @ matrix, torch.eye(3).expand(5, 3, 3), atol=1e-5)
    assert torch.allclose(matrix.det(), torch.ones(5), atol=1e-5)


def test_matrix_to_axis_angle_z_rotation() -> None:
    axis_angle = matrix_to_axis_angle(_rot_z(math.pi / 2))
    assert torch.allclose(axis_angle, torch.tensor([0.0, 0.0, math.pi / 2]), atol=1e-5)


def test_geodesic_distance_identity_and_z90() -> None:
    distance = geodesic_distance(torch.eye(3), _rot_z(math.pi / 2))
    assert torch.allclose(distance, torch.tensor(math.pi / 2), atol=1e-5)


def test_geodesic_distance_same_is_zero() -> None:
    matrix = _rot_z(0.3).expand(4, 3, 3)
    distance = geodesic_distance(matrix, matrix)
    assert torch.allclose(distance, torch.zeros(4), atol=1e-5)


def test_identity_rotation_log_and_geodesic_have_finite_gradients() -> None:
    identity = torch.eye(3, requires_grad=True)
    axis_angle = matrix_to_axis_angle(identity)
    distance = geodesic_distance(identity, torch.eye(3))
    (axis_angle.square().sum() + distance).backward()
    assert torch.isfinite(identity.grad).all()


def test_matrix_to_axis_angle_pi_rotation_is_finite() -> None:
    matrix = _rot_z(math.pi).requires_grad_()
    axis_angle = matrix_to_axis_angle(matrix)
    axis_angle.sum().backward()
    assert torch.isfinite(axis_angle).all()
    assert torch.isfinite(matrix.grad).all()
    assert torch.allclose(axis_angle.abs(), torch.tensor([0.0, 0.0, math.pi]), atol=1e-5)
