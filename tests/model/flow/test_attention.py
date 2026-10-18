from __future__ import annotations

import torch

from src.model.flow.attention import CrossAttention, Modulation, SelfAttention, apply_rotary


def test_apply_rotary_shape_and_orthogonality() -> None:
    b, h, s, d = 2, 4, 6, 8
    value = torch.randn(b, h, s, d)
    positions = torch.arange(s).float().expand(b, s)
    rotated = apply_rotary(value, positions)
    assert rotated.shape == value.shape
    # RoPE preserves per-head, per-position norm
    assert torch.allclose(rotated.norm(dim=-1), value.norm(dim=-1), atol=1e-5)


def test_self_attention_shape_and_mask() -> None:
    attn = SelfAttention(16, 4)
    x = torch.randn(2, 5, 16)
    positions = torch.arange(5).float().expand(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, -2:] = False
    out = attn(x, x, x, positions, positions, mask)
    assert out.shape == (2, 5, 16)
    assert out.isfinite().all()


def test_self_attention_joint_two_lengths() -> None:
    attn = SelfAttention(16, 4)
    q = torch.randn(2, 6, 16)
    kv = torch.randn(2, 9, 16)
    q_pos = torch.arange(6).float().expand(2, 6)
    kv_pos = torch.arange(9).float().expand(2, 9)
    mask = torch.ones(2, 9, dtype=torch.bool)
    out = attn(q, kv, kv, q_pos, kv_pos, mask)
    assert out.shape == (2, 6, 16)


def test_self_attention_latent_queries_with_self_and_condition_kv() -> None:
    attn = SelfAttention(16, 4)
    query = torch.randn(2, 6, 16)
    condition = torch.randn(2, 18, 16)
    key_value = torch.cat([query, condition], dim=1)
    query_positions = torch.arange(6).float().expand(2, 6)
    condition_positions = torch.arange(18).float().expand(2, 18)
    key_positions = torch.cat([query_positions, condition_positions], dim=1)
    out = attn(query, key_value, key_value, query_positions, key_positions)
    assert out.shape == query.shape
    assert out.isfinite().all()


def test_cross_attention_shape() -> None:
    attn = CrossAttention(16, 4)
    q = torch.randn(2, 6, 16)
    kv = torch.randn(2, 30, 16)
    q_pos = torch.arange(6).float().expand(2, 6)
    kv_pos = torch.arange(30).float().expand(2, 30)
    out = attn(q, kv, q_pos, kv_pos)
    assert out.shape == (2, 6, 16)


def test_modulation_zero_init() -> None:
    mod = Modulation(16, 3)
    emb = torch.randn(2, 16)
    shift, scale, gate = mod(emb)
    assert shift.shape == (2, 3, 16)
    assert scale.shape == (2, 3, 16)
    assert gate.shape == (2, 3, 16)
    # zero-init: all zeros initially
    assert (shift == 0).all() and (scale == 0).all() and (gate == 0).all()
