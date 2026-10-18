from __future__ import annotations

import torch

from src.model.flow.joint import JointFlowModel, JointFlowState


def test_joint_flow_model_shapes_and_gradients() -> None:
    model = JointFlowModel(model_dim=32, depth=1, heads=4)
    state = JointFlowState(torch.randn(2, 2, 4, 16, 3))
    output = model(state, torch.rand(2))
    assert output.joints.shape == state.joints.shape
    assert output.joints.isfinite().all()
    output.joints.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_joint_flow_model_rejects_wrong_shape() -> None:
    model = JointFlowModel(model_dim=32, depth=1, heads=4)
    try:
        model(JointFlowState(torch.zeros(1, 2, 3, 15, 3)), torch.zeros(1))
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("wrong joint count must fail")
