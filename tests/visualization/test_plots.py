from __future__ import annotations

import numpy as np
import pytest
import torch

from src.visualization import (
    plot_projected_mesh_sequence,
    plot_trajectory_comparison,
    project_points,
    transform_vertices_to_camera,
)


def _identity_6d(*shape):
    return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(*shape, 6).clone()


def test_transform_vertices_to_camera_inverts_generated_reference_pose():
    vertices = torch.zeros(1, 2, 2, 1, 3)
    vertices[..., 0, :] = torch.tensor([2.0, 4.0, 6.0])
    rotation = _identity_6d(1, 2)
    translation = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])

    result = transform_vertices_to_camera(vertices, rotation, translation)

    assert result.shape == vertices.shape
    assert torch.allclose(result[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(result[0, :, 1, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_project_points_returns_uv_depth_and_visibility():
    points = torch.tensor(
        [[[[[1.0, 2.0, 2.0], [0.0, 0.0, -1.0]]], [[ [2.0, 0.0, 2.0], [0.0, 1.0, 2.0]]]]]
    )
    intrinsics = torch.tensor([[100.0, 0.0, 10.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]])

    uv, depth, valid = project_points(points, intrinsics)

    assert uv.shape == (1, 2, 1, 2, 2)
    assert torch.allclose(uv[0, 0, 0, 0], torch.tensor([60.0, 120.0]))
    assert depth[0, 0, 0, 1] < 0
    assert valid[0, 0, 0, 0]
    assert not valid[0, 0, 0, 1]


def test_geometry_accepts_numpy_inputs():
    vertices = np.zeros((2, 1, 1, 3), dtype=np.float32)
    rotation = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (1, 1))
    translation = np.zeros((1, 3), dtype=np.float32)
    result = transform_vertices_to_camera(vertices, rotation, translation)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 2, 1, 1, 3)


@pytest.mark.parametrize("bad_shape", [torch.zeros(1, 3, 2, 4), torch.zeros(1, 2, 2, 4, 2)])
def test_transform_rejects_invalid_vertex_shape(bad_shape):
    with pytest.raises(ValueError):
        transform_vertices_to_camera(bad_shape, _identity_6d(1, 2), torch.zeros(1, 2, 3))


def test_plots_create_trajectory_and_mesh_artists():
    pytest.importorskip("matplotlib")
    joints = torch.zeros(1, 2, 3, 16, 3)
    joints[0, 0, :, 0, 0] = torch.arange(3)
    joints[0, 1, :, 0, 1] = torch.arange(3)
    camera = torch.zeros(1, 3, 3)
    figure, axes = plot_trajectory_comparison(
        joints, joints, camera, camera, metrics={"mpjpe_mm": 0.0}, title="test"
    )
    assert len(axes[0].lines) == 4
    assert len(axes[1].lines) == 2
    assert any("mpjpe_mm" in text.get_text() for text in figure.texts)

    vertices = torch.tensor(
        [[[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.5, 0.5, 2.0], [-0.5, 0.5, 2.0]]]
    ).unsqueeze(1).expand(1, 2, 3, 4, 3)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    intrinsics = torch.eye(3).repeat(3, 1, 1)
    mesh_figure, mesh_axes = plot_projected_mesh_sequence(
        vertices, faces, intrinsics, frame_indices=[0, 2], image_size=(10, 10)
    )
    assert mesh_axes.shape == (2, 2)
    assert all(axis.collections for axis in mesh_axes.flat)
    mesh_figure.clf()
    figure.clf()
