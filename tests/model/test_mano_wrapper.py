from __future__ import annotations

import torch

from src.loss.losses import acceleration_loss, joints3d_loss, camera_pose_loss
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper


def _mano(requires_grad: bool = True) -> ManoFlowState:
    return ManoFlowState(
        translation=torch.randn(1, 2, 3, 3, requires_grad=requires_grad),
        root_rotation=torch.randn(1, 2, 3, 6, requires_grad=requires_grad),
        local_rotation=torch.randn(1, 2, 3, 15, 6, requires_grad=requires_grad),
    )


def test_real_mano_joints_and_acceleration_keep_input_gradients() -> None:
    prediction = _mano()
    target = _mano(False)
    wrapper = ManoWrapper()

    joints_loss = joints3d_loss(prediction, target, wrapper)
    acceleration = acceleration_loss(prediction, target, wrapper)
    (joints_loss + acceleration).backward()

    assert joints_loss.isfinite()
    assert acceleration.isfinite()
    for value in (prediction.translation, prediction.root_rotation, prediction.local_rotation):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
    assert set(wrapper._layers) == {"left", "right"}


def test_real_mano_lazy_layers_follow_input_dtype() -> None:
    wrapper = ManoWrapper()
    state = _mano().map(lambda value: value.to(dtype=torch.float64))
    joints = wrapper(state)
    assert joints.dtype == torch.float64
    assert all(parameter.dtype == torch.float64 for parameter in wrapper.parameters())


def test_camera_pose_identity_gradient_is_finite() -> None:
    rotation = torch.tensor([1., 0., 0., 0., 1., 0.]).expand(1, 1, 6).clone().requires_grad_()
    state = CameraFlowState(rotation=rotation, translation=torch.zeros(1, 1, 3, requires_grad=True))
    target = CameraFlowState(rotation=rotation.detach().clone(), translation=torch.zeros(1, 1, 3))
    loss = camera_pose_loss(state, target)
    loss.backward()
    assert loss.isfinite()
    assert torch.isfinite(rotation.grad).all()
