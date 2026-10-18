"""Flow parameter sampling: Gaussian noise and time step."""

from __future__ import annotations

import torch
from torch import Tensor

from src.model.flow.state import CameraFlowState, ManoFlowState


def sample_noise(
    target_mano: ManoFlowState,
    target_camera: CameraFlowState,
    generator: torch.Generator | None = None,
) -> tuple[ManoFlowState, CameraFlowState]:
    """Sample Gaussian noise with the same shape as the targets."""
    return sample_noise_shapes(
        target_mano.translation.shape,
        target_mano.root_rotation.shape,
        target_mano.local_rotation.shape,
        target_camera.rotation.shape,
        target_camera.translation.shape,
        device=target_mano.translation.device,
        dtype=target_mano.translation.dtype,
        generator=generator,
    )


def sample_noise_shapes(
    translation_shape,
    root_rotation_shape,
    local_rotation_shape,
    camera_rotation_shape,
    camera_translation_shape,
    *,
    device,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[ManoFlowState, CameraFlowState]:
    """Sample latent shapes without requiring any supervision target."""
    def draw(shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    return (
        ManoFlowState(draw(translation_shape), draw(root_rotation_shape), draw(local_rotation_shape)),
        CameraFlowState(draw(camera_rotation_shape), draw(camera_translation_shape)),
    )


def sample_flow_time(
    batch: int,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``t = sigmoid(z)`` with ``z ~ N(0, sigma)``."""
    device = generator.device if generator is not None else None
    z = torch.randn(batch, device=device, generator=generator) * sigma
    return torch.sigmoid(z)
