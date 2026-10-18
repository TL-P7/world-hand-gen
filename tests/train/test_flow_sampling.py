from __future__ import annotations

import torch

from src.model.flow.state import CameraFlowState, ManoFlowState
from src.train.flow_sampling import sample_flow_time, sample_noise


def _mano() -> ManoFlowState:
    return ManoFlowState(
        translation=torch.zeros(2, 2, 4, 3),
        root_rotation=torch.zeros(2, 2, 4, 6),
        local_rotation=torch.zeros(2, 2, 4, 15, 6),
    )


def _camera() -> CameraFlowState:
    return CameraFlowState(
        rotation=torch.zeros(2, 4, 6),
        translation=torch.zeros(2, 4, 3),
    )


def test_sample_noise_shape_matches_target() -> None:
    noise_mano, noise_camera = sample_noise(_mano(), _camera())
    assert noise_mano.translation.shape == (2, 2, 4, 3)
    assert noise_mano.local_rotation.shape == (2, 2, 4, 15, 6)
    assert noise_camera.rotation.shape == (2, 4, 6)


def test_sample_noise_is_gaussian_and_finite() -> None:
    noise_mano, _ = sample_noise(_mano(), _camera())
    assert noise_mano.translation.isfinite().all()
    # mean ~ 0, std ~ 1 for a large enough sample
    assert noise_mano.translation.abs().mean() < 2.0


def test_flow_time_in_unit_interval() -> None:
    t = sample_flow_time(1000, sigma=1.0)
    assert (t >= 0).all() and (t <= 1).all()


def test_flow_time_smaller_sigma_concentrates_middle() -> None:
    wide = sample_flow_time(10000, sigma=1.0)
    narrow = sample_flow_time(10000, sigma=0.3)
    # Smaller sigma concentrates toward 0.5: smaller deviation from 0.5.
    assert (narrow - 0.5).abs().mean() < (wide - 0.5).abs().mean()


def test_flow_time_supports_cuda_generator() -> None:
    if not torch.cuda.is_available():
        return
    generator = torch.Generator(device="cuda").manual_seed(0)
    t = sample_flow_time(16, sigma=0.9, generator=generator)
    assert t.device.type == "cuda"
    assert torch.isfinite(t).all()
    assert (t >= 0).all() and (t <= 1).all()
