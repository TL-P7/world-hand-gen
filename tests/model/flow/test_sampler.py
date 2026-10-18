from __future__ import annotations

import torch

from src.model.flow.mmdit import MMDiT
from src.model.flow.sampler import sample, sample_noise_state
from src.model.flow.state import ConditionTokens


def _conditions(batch, time, dim):
    return ConditionTokens(
        hamer=torch.randn(batch, 2 * time, dim),
        vggt=torch.randn(batch, 17 * time, dim),
        bbox=torch.randn(batch, 2 * time, dim),
        keypoints=torch.randn(batch, 2 * time, dim),
    )


def test_sample_noise_state_shapes() -> None:
    mano, camera = sample_noise_state(batch=2, time=4, device=torch.device("cpu"))
    assert mano.translation.shape == (2, 2, 4, 3)
    assert mano.local_rotation.shape == (2, 2, 4, 15, 6)
    assert camera.rotation.shape == (2, 4, 6)
    assert camera.translation.shape == (2, 4, 3)


def test_sample_runs_and_returns_shapes() -> None:
    model = MMDiT(model_dim=16, depth=1, heads=2)
    conditions = _conditions(1, 3, 16)
    mano, camera = sample(
        model,
        conditions,
        batch=1,
        time=3,
        steps=10,
        device=torch.device("cpu"),
    )
    assert mano.translation.shape == (1, 2, 3, 3)
    assert camera.rotation.shape == (1, 3, 6)
    assert mano.translation.isfinite().all()
