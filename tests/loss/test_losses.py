from __future__ import annotations

import math

import torch

from src.loss.losses import (
    acceleration_loss,
    camera_flow_loss,
    camera_pose_loss,
    camera_trajectory_loss,
    flow_loss,
    joints3d_loss,
)
from src.geometry.rotation import rotation_6d_to_matrix
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper


IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class _FakeManoWrapper:
    """Returns joints equal to translation so loss logic is testable offline."""

    def __call__(self, mano: ManoFlowState) -> torch.Tensor:
        return mano.translation.unsqueeze(3).expand(-1, -1, -1, 16, 3)


def _mano(value: float) -> ManoFlowState:
    return ManoFlowState(
        translation=torch.full((1, 2, 4, 3), value),
        root_rotation=torch.full((1, 2, 4, 6), value),
        local_rotation=torch.full((1, 2, 4, 15, 6), value),
    )


def _camera(translation: float = 0.0) -> CameraFlowState:
    return CameraFlowState(
        rotation=IDENTITY_6D.expand(1, 4, 6).clone(),
        translation=torch.full((1, 4, 3), translation),
    )


def test_flow_loss_is_direct_mean_mse() -> None:
    assert flow_loss(_mano(0.0), _mano(1.0)) == torch.tensor(1.0)
    assert torch.allclose(camera_flow_loss(_camera(0.0), _camera(1.0)), torch.tensor(1.0 / 3.0))


def test_camera_pose_loss_identity() -> None:
    assert camera_pose_loss(_camera(), _camera()) == 0


def test_joints3d_loss_uses_mano_wrapper() -> None:
    mano = _FakeManoWrapper()
    assert joints3d_loss(_mano(0.0), _mano(1.0), mano) == 1


def test_acceleration_loss_uses_mano_wrapper() -> None:
    mano = _FakeManoWrapper()
    t = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1).expand(1, 2, 6, 3)
    constant_velocity = ManoFlowState(
        translation=t, root_rotation=torch.zeros(1, 2, 6, 6), local_rotation=torch.zeros(1, 2, 6, 15, 6)
    )
    assert acceleration_loss(constant_velocity, constant_velocity, mano) == 0


def test_camera_trajectory_loss_identity() -> None:
    assert camera_trajectory_loss(_camera(), _camera()) == 0


def test_camera_pose_rotation_is_geodesic() -> None:
    angle = math.pi / 2
    c, s = math.cos(angle), math.sin(angle)
    matrix = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotation = torch.cat((matrix[:, 0], matrix[:, 1]))
    prediction = CameraFlowState(rotation=rotation.reshape(1, 1, 6), translation=torch.zeros(1, 1, 3))
    target = _camera().map(lambda x: x[:, :1])
    assert torch.allclose(camera_pose_loss(prediction, target), torch.tensor(angle), atol=1e-5)


def test_rotation_6d_output_is_so3() -> None:
    assert torch.allclose(
        rotation_6d_to_matrix(torch.randn(4, 6)).det(), torch.ones(4), atol=1e-5
    )
