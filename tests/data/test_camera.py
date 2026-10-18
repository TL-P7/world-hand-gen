from __future__ import annotations

import torch

from src.data.camera import (
    TARGET_CAMERA_SEMANTICS,
    camera_flow_from_extrinsics,
    relative_camera_transforms,
    world_points_to_first_camera,
)
from src.geometry.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix


def _pose(rotation: torch.Tensor, translation: tuple[float, float, float]) -> torch.Tensor:
    result = torch.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = torch.tensor(translation)
    return result


def test_relative_camera_transform_uses_current_to_first_direction() -> None:
    quarter_turn = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    first = _pose(quarter_turn, (2.0, 3.0, 4.0))
    current = _pose(torch.eye(3), (5.0, 3.0, 4.0))
    extrinsics = torch.stack((first, current))
    frame_transform = torch.eye(4)
    frame_transform[:3, :3] = quarter_turn

    expected = frame_transform @ torch.linalg.inv(first) @ current @ torch.linalg.inv(frame_transform)
    actual = relative_camera_transforms(extrinsics, frame_transform)

    torch.testing.assert_close(actual, torch.stack((torch.eye(4), expected)))
    assert torch.allclose(actual[0, :3, 3], torch.zeros(3))


def test_camera_flow_round_trip_and_point_frame_share_semantics() -> None:
    first = _pose(torch.eye(3), (10.0, 0.0, 0.0))
    current = _pose(torch.eye(3), (12.0, 0.0, 0.0))
    extrinsics = torch.stack((first, current))
    flow = camera_flow_from_extrinsics(extrinsics)
    expected_rotation = matrix_to_rotation_6d(torch.eye(3)).expand(2, -1)

    assert TARGET_CAMERA_SEMANTICS == "current_model_camera_to_first_model_camera"
    torch.testing.assert_close(flow.rotation[0], expected_rotation[0])
    torch.testing.assert_close(flow.translation, torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    torch.testing.assert_close(rotation_6d_to_matrix(flow.rotation), torch.eye(3).expand(2, 3, 3))

    points = torch.tensor([[[12.0, 1.0, 0.0], [12.0, 0.0, 2.0]]])
    transformed = world_points_to_first_camera(points, first)
    torch.testing.assert_close(transformed, torch.tensor([[[2.0, 1.0, 0.0], [2.0, 0.0, 2.0]]]))
