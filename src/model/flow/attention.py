"""Attention primitives with RoPE and AdaLN-Zero.

- ``apply_rotary`` rotates queries/keys by frame positions.
- ``SelfAttention`` is a multi-head attention over a shared token set; the
  joint case is implemented by calling it with concatenated MANO+Camera tokens.
- ``CrossAttention`` attends from a latent query stream to condition key/value.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def apply_rotary(value: Tensor, positions: Tensor) -> Tensor:
    """Rotate ``value`` ``[B,H,S,D]`` by frame positions ``[B,S]``."""
    _, _, _, dim = value.shape
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, device=value.device, dtype=value.dtype)
        / dim
    )
    angles = positions.unsqueeze(-1) * frequencies  # [B,S,D/2]
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Modulation(nn.Module):
    """Time-conditioned AdaLN-Zero: maps a flow embedding to shift/scale/gate.

    ``num`` independent (shift, scale, gate) triplets are produced for a single
    stream, so one module can serve several sub-layers of a block.
    """

    def __init__(self, model_dim: int, num: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num = num
        self.linear = nn.Linear(model_dim, 3 * num * model_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, flow_embedding: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(shift, scale, gate)``, each ``[B, num, model_dim]``."""
        out = self.linear(flow_embedding)
        return out.reshape(-1, self.num, 3, self.model_dim).unbind(dim=2)


def modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN shift/scale: ``value * (1+scale) + shift`` over sequence axis."""
    return _modulate(value, shift, scale)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE on queries and keys."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        query: Tensor,  # [B,S_q,D]
        key: Tensor,  # [B,S_k,D]
        value: Tensor,  # [B,S_k,D]
        query_positions: Tensor,  # [B,S_q]
        key_positions: Tensor,  # [B,S_k]
        key_mask: Tensor | None = None,  # bool [B,S_k]
    ) -> Tensor:
        batch = query.shape[0]
        q = self.q_proj(query).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, query_positions)
        k = apply_rotary(k, key_positions)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            has_key = key_mask.any(dim=-1)  # [B]
            scores = torch.where(has_key[:, None, None, None], scores, torch.zeros_like(scores))
            attn = scores.softmax(dim=-1)
            attn = attn * has_key[:, None, None, None].float()
        else:
            attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, -1, self.model_dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention from latent queries to condition key/value."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        query: Tensor,  # [B,S_q,D]
        key_value: Tensor,  # [B,S_c,D]
        query_positions: Tensor,  # [B,S_q]
        key_positions: Tensor,  # [B,S_c]
    ) -> Tensor:
        batch = query.shape[0]
        q = self.q_proj(query).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, query_positions)
        k = apply_rotary(k, key_positions)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, -1, self.model_dim)
        return self.out_proj(out)
