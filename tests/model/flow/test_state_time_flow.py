from __future__ import annotations

import pytest
import torch

from src.model.flow.rectified_flow import flow_interpolate, flow_velocity_from_x0
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.flow.time import ScalarTimeEncoder, sinusoidal_embedding


def _mano(scale: float = 1.0) -> ManoFlowState:
    return ManoFlowState(
        translation=torch.full((2, 2, 4, 3), scale),
        root_rotation=torch.full((2, 2, 4, 6), scale),
        local_rotation=torch.full((2, 2, 4, 15, 6), scale),
    )


def _camera(scale: float = 1.0) -> CameraFlowState:
    return CameraFlowState(
        rotation=torch.full((2, 4, 6), scale),
        translation=torch.full((2, 4, 3), scale),
    )


@pytest.mark.parametrize("state", [_mano(), _camera()])
def test_interpolate_endpoints(state) -> None:
    t = torch.tensor([0.0, 1.0])
    noise = state.map(lambda x: torch.zeros_like(x))
    result = flow_interpolate(noise, state, t)
    for field in ("translation", "rotation", "root_rotation", "local_rotation"):
        if hasattr(result, field):
            value = getattr(result, field)
            target = getattr(state, field)
            assert torch.allclose(value, target * t.reshape((-1,) + (1,) * (value.ndim - 1)))


def test_interpolate_midpoint() -> None:
    noise = _mano(0.0)
    target = _mano(4.0)
    result = flow_interpolate(noise, target, torch.tensor([0.5]))
    assert torch.allclose(result.translation, torch.tensor([2.0]))
    assert torch.allclose(result.local_rotation, torch.tensor([2.0]))


def test_velocity_matches_x0() -> None:
    # state = (1-t)noise + t*x0; velocity should recover (x0 - state)/(1-t)
    x0 = _mano(3.0)
    noise = _mano(-1.0)
    t = torch.tensor([0.25])
    state = flow_interpolate(noise, x0, t)
    velocity = flow_velocity_from_x0(state, x0, t)
    # state = 0.75*(-1) + 0.25*3 = 0; velocity = (3 - 0)/0.75 = 4
    assert torch.allclose(velocity.translation, torch.tensor([4.0]))


def test_velocity_rejects_t_one() -> None:
    with pytest.raises(ValueError, match="strictly less than 1"):
        flow_velocity_from_x0(_mano(), _mano(), torch.tensor([1.0]))


def test_sinusoidal_embedding_shape_and_period() -> None:
    values = torch.tensor([0.0, 0.5])
    emb = sinusoidal_embedding(values, 64)
    assert emb.shape == (2, 64)
    assert torch.allclose(emb[0, :32], torch.tensor([1.0] * 32))  # cos(0)=1


def test_time_encoder_output_shape() -> None:
    encoder = ScalarTimeEncoder(512)
    out = encoder(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, 512)
    assert out.isfinite().all()
