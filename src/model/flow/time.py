"""Flow-time encoding and sinusoidal embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_embedding(values: Tensor, dimension: int, max_period: float = 10000.0) -> Tensor:
    """Embed scalar values into ``dimension`` sinusoidal features."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device, dtype=values.dtype) / max(half - 1, 1)
    )
    angles = values.unsqueeze(-1) * frequencies
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[..., :1])), dim=-1)
    return embedding


class ScalarTimeEncoder(nn.Module):
    """Encode a scalar flow time into a ``model_dim`` vector."""

    def __init__(self, model_dim: int, fourier_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(fourier_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(sinusoidal_embedding(values, self.network[0].in_features))
