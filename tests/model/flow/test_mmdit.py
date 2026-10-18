from __future__ import annotations

import torch

from src.model.flow.condition import ConditionEncoder
from src.model.flow.mmdit import MMDiT
from src.model.flow.state import CameraFlowState, ManoFlowState


def _conditions(b, t, model_dim):
    encoder = ConditionEncoder(hand_dim=1024, camera_dim=2048, model_dim=model_dim)
    hamer = torch.randn(b, 2, t, 1024)
    vggt = torch.randn(b, t, 17, 2048)
    bbox = torch.randn(b, 2, t, 4)
    keypoints = torch.rand(b, 2, t, 21, 3)
    masks = (
        torch.ones(b, 2, t, dtype=torch.bool),
        torch.ones(b, t, dtype=torch.bool),
        torch.ones(b, 2, t, dtype=torch.bool),
    )
    return encoder(hamer, vggt, bbox, *masks, keypoints)


def _model(model_dim=32, depth=2, heads=4):
    return MMDiT(model_dim=model_dim, depth=depth, heads=heads)


def test_mmdit_forward_shapes() -> None:
    b, t = 2, 4
    model = _model()
    mano = ManoFlowState(
        translation=torch.randn(b, 2, t, 3),
        root_rotation=torch.randn(b, 2, t, 6),
        local_rotation=torch.randn(b, 2, t, 15, 6),
    )
    camera = CameraFlowState(rotation=torch.randn(b, t, 6), translation=torch.randn(b, t, 3))
    conditions = _conditions(b, t, 32)
    flow_time = torch.rand(b)

    mano_out, camera_out = model(mano, camera, flow_time, conditions)

    assert mano_out.translation.shape == (b, 2, t, 3)
    assert mano_out.root_rotation.shape == (b, 2, t, 6)
    assert mano_out.local_rotation.shape == (b, 2, t, 15, 6)
    assert camera_out.rotation.shape == (b, t, 6)
    assert camera_out.translation.shape == (b, t, 3)
    assert all(x.isfinite().all() for x in (mano_out.translation, mano_out.root_rotation, mano_out.local_rotation, camera_out.rotation, camera_out.translation))


def test_mmdit_identity_rotation_at_init() -> None:
    b, t = 1, 2
    model = _model()
    mano = ManoFlowState(
        translation=torch.zeros(b, 2, t, 3),
        root_rotation=torch.zeros(b, 2, t, 6),
        local_rotation=torch.zeros(b, 2, t, 15, 6),
    )
    camera = CameraFlowState(rotation=torch.zeros(b, t, 6), translation=torch.zeros(b, t, 3))
    conditions = _conditions(b, t, 32)
    flow_time = torch.zeros(b)

    mano_out, camera_out = model(mano, camera, flow_time, conditions)

    # 6D identity = [1,0,0,0,1,0] on the rotation coordinates (translation head still 0)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert torch.allclose(mano_out.root_rotation, expected.expand_as(mano_out.root_rotation))
    assert torch.allclose(camera_out.rotation, expected.expand_as(camera_out.rotation))


def test_mmdit_backward_propagates() -> None:
    b, t = 1, 2
    model = _model()
    mano = ManoFlowState(
        translation=torch.randn(b, 2, t, 3, requires_grad=True),
        root_rotation=torch.randn(b, 2, t, 6, requires_grad=True),
        local_rotation=torch.randn(b, 2, t, 15, 6, requires_grad=True),
    )
    camera = CameraFlowState(
        rotation=torch.randn(b, t, 6, requires_grad=True),
        translation=torch.randn(b, t, 3, requires_grad=True),
    )
    conditions = _conditions(b, t, 32)
    flow_time = torch.rand(b)

    mano_out, camera_out = model(mano, camera, flow_time, conditions)
    loss = mano_out.translation.sum() + camera_out.rotation.sum()
    loss.backward()
    assert mano.translation.grad is not None
    assert camera.rotation.grad is not None
    assert mano.translation.grad.isfinite().all()
