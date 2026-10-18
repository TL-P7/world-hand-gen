"""Heun ODE sampler for the flow-matching MM-DiT.

Integrates from ``t=0`` (pure noise) to ``t=1`` (clean prediction) by repeated
model evaluation. The result is the sampled MANO and camera state in 6D flow
space; callers convert 6D back to matrices for downstream use.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .mmdit import MMDiT
from .rectified_flow import flow_velocity_from_x0
from .state import CameraFlowState, ConditionTokens, ManoFlowState


def sample_noise_state(
    batch: int,
    time: int,
    device: torch.device,
    *,
    local_joints: int = 15,
    generator: torch.Generator | None = None,
) -> tuple[ManoFlowState, CameraFlowState]:
    """Sample initial Gaussian noise for MANO and camera in flow space."""
    mano = ManoFlowState(
        translation=torch.randn(batch, 2, time, 3, device=device, generator=generator),
        root_rotation=torch.randn(batch, 2, time, 6, device=device, generator=generator),
        local_rotation=torch.randn(
            batch, 2, time, local_joints, 6, device=device, generator=generator
        ),
    )
    camera = CameraFlowState(
        rotation=torch.randn(batch, time, 6, device=device, generator=generator),
        translation=torch.randn(batch, time, 3, device=device, generator=generator),
    )
    return mano, camera


def _add_state_scaled(state, delta, scale):
    return type(state)(
        **{
            name: getattr(state, name) + scale * getattr(delta, name)
            for name in state.__dataclass_fields__
        }
    )


@torch.no_grad()
def sample(
    model: MMDiT,
    conditions: ConditionTokens,
    *,
    batch: int,
    time: int,
    steps: int = 10,
    device: torch.device,
    noise: tuple[ManoFlowState, CameraFlowState] | None = None,
) -> tuple[ManoFlowState, CameraFlowState]:
    """Run Heun integration from noise to a clean sample.

    Each step evaluates the model twice (predictor + corrector), except the
    final step which uses a single Euler update, giving ``nfe = 2*steps - 1``.
    """
    mano, camera = noise or sample_noise_state(batch, time, device)
    step_size = 1.0 / steps

    for index in range(steps):
        flow_time = torch.full((batch,), index / steps, device=device)
        first_mano, first_camera = model(mano, camera, flow_time, conditions)
        v_mano = flow_velocity_from_x0(mano, first_mano, flow_time)
        v_camera = flow_velocity_from_x0(camera, first_camera, flow_time)

        if index == steps - 1:
            mano = _add_state_scaled(mano, v_mano, step_size)
            camera = _add_state_scaled(camera, v_camera, step_size)
            break

        next_time = torch.full((batch,), (index + 1) / steps, device=device)
        predictor_mano = _add_state_scaled(mano, v_mano, step_size)
        predictor_camera = _add_state_scaled(camera, v_camera, step_size)

        second_mano, second_camera = model(
            predictor_mano, predictor_camera, next_time, conditions
        )
        v2_mano = flow_velocity_from_x0(predictor_mano, second_mano, next_time)
        v2_camera = flow_velocity_from_x0(predictor_camera, second_camera, next_time)

        mano = _add_state_scaled(mano, v_mano, 0.5 * step_size)
        camera = _add_state_scaled(camera, v_camera, 0.5 * step_size)
        mano = _add_state_scaled(mano, v2_mano, 0.5 * step_size)
        camera = _add_state_scaled(camera, v2_camera, 0.5 * step_size)

    return mano, camera
