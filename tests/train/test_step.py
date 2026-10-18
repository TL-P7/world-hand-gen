from __future__ import annotations

import torch
from torch import nn

from src.model.flow.condition import ConditionEncoder
from src.model.flow.mmdit import MMDiT
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.train.flow_sampling import sample_noise
from src.train.step import train_step


class _FakeManoWrapper:
    def __call__(self, mano: ManoFlowState) -> torch.Tensor:
        return mano.translation.unsqueeze(3).expand(-1, -1, -1, 16, 3)


def _mano(batch=1, t=4) -> ManoFlowState:
    return ManoFlowState(
        translation=torch.randn(batch, 2, t, 3),
        root_rotation=torch.randn(batch, 2, t, 6),
        local_rotation=torch.randn(batch, 2, t, 15, 6),
    )


def _camera(batch=1, t=4) -> CameraFlowState:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return CameraFlowState(
        rotation=identity.expand(batch, t, 6).clone(),
        translation=torch.randn(batch, t, 3),
    )


def _conditions(batch=1, t=4, dim=32):
    encoder = ConditionEncoder(hand_dim=1024, camera_dim=2048, model_dim=dim)
    hamer = torch.randn(batch, 2, t, 1024)
    vggt = torch.randn(batch, t, 17, 2048)
    bbox = torch.randn(batch, 2, t, 4)
    keypoints = torch.rand(batch, 2, t, 21, 3)
    masks = (
        torch.ones(batch, 2, t, dtype=torch.bool),
        torch.ones(batch, t, dtype=torch.bool),
        torch.ones(batch, 2, t, dtype=torch.bool),
    )
    return encoder(hamer, vggt, bbox, *masks, keypoints)


def test_train_step_returns_total_and_terms() -> None:
    model = MMDiT(model_dim=32, depth=1, heads=4)
    total, terms = train_step(
        model=model,
        mano_wrapper=_FakeManoWrapper(),
        target_mano=_mano(),
        target_camera=_camera(),
        noise_mano=sample_noise(_mano(), _camera())[0],
        noise_camera=sample_noise(_mano(), _camera())[1],
        conditions=_conditions(dim=32),
        flow_time=torch.rand(1),
    )
    assert total.shape == ()
    assert total.isfinite()
    assert "flow" in terms and "camera_flow" in terms
    assert "joints3d" in terms and "camera_pose" in terms


def test_train_step_auxiliary_gate() -> None:
    model = MMDiT(model_dim=32, depth=1, heads=4)
    total, terms = train_step(
        model=model,
        mano_wrapper=_FakeManoWrapper(),
        target_mano=_mano(),
        target_camera=_camera(),
        noise_mano=sample_noise(_mano(), _camera())[0],
        noise_camera=sample_noise(_mano(), _camera())[1],
        conditions=_conditions(dim=32),
        flow_time=torch.rand(1),
        auxiliary_enabled=False,
    )
    assert "flow" in terms and "camera_flow" in terms
    assert "joints3d" not in terms
    assert "camera_pose" not in terms


def test_train_step_gradient_flows() -> None:
    model = MMDiT(model_dim=32, depth=1, heads=4)
    total, _ = train_step(
        model=model,
        mano_wrapper=_FakeManoWrapper(),
        target_mano=_mano(),
        target_camera=_camera(),
        noise_mano=sample_noise(_mano(), _camera())[0],
        noise_camera=sample_noise(_mano(), _camera())[1],
        conditions=_conditions(dim=32),
        flow_time=torch.rand(1),
    )
    total.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(p.grad is not None for p in params)
