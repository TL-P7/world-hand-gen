from __future__ import annotations

import torch

from src.model.flow.joint import JointFlowModel, JointFlowState
from src.train.joint_flow import loss, train_joint_step


def test_masked_joint_loss_ignores_invalid_points() -> None:
    prediction = JointFlowState(torch.zeros(1, 2, 1, 16, 3))
    target = JointFlowState(torch.ones(1, 2, 1, 16, 3))
    valid = torch.ones(1, 2, 1, 16, dtype=torch.bool)
    valid[..., 1:] = False
    target.joints[..., 1:, :] = float("nan")
    assert loss(prediction, target, valid).item() == 1.0


def test_joint_train_step_backpropagates_without_mano_or_camera() -> None:
    model = JointFlowModel(model_dim=32, depth=1, heads=4)
    target = JointFlowState(torch.randn(1, 2, 4, 16, 3))
    valid = torch.ones(1, 2, 4, 16, dtype=torch.bool)
    noise = JointFlowState(torch.randn_like(target.joints))
    value, prediction = train_joint_step(model, target, valid, torch.full((1,), 0.5), noise)
    assert value.isfinite()
    value.backward()
    assert prediction.joints.shape == target.joints.shape
    assert any(parameter.grad is not None for parameter in model.parameters())
