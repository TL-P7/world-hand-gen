from __future__ import annotations

import torch

from src.model.flow.mmdit import MMDiTBlock


def _block() -> MMDiTBlock:
    return MMDiTBlock(model_dim=32, heads=4)


def _conditions(batch: int, time: int, model_dim: int):
    hamer = torch.randn(batch, 2 * time, model_dim)
    vggt = torch.randn(batch, 17 * time, model_dim)
    bbox = torch.randn(batch, 2 * time, model_dim)
    frame = torch.arange(time).float()
    mano_pos = frame.repeat(2).expand(batch, 2 * time)
    camera_pos = frame.expand(batch, time)
    vggt_pos = frame.repeat_interleave(17).expand(batch, 17 * time)
    condition = torch.cat((hamer, vggt, bbox), dim=1)
    condition_pos = torch.cat((mano_pos, vggt_pos, mano_pos), dim=1)
    return condition, condition_pos, hamer, vggt, bbox, mano_pos, camera_pos, vggt_pos


def _forward(block: MMDiTBlock, mano: torch.Tensor, camera: torch.Tensor, emb: torch.Tensor):
    batch, _, model_dim = mano.shape
    time = camera.shape[1]
    (
        condition,
        condition_pos,
        hamer,
        vggt,
        bbox,
        mano_pos,
        camera_pos,
        vggt_pos,
    ) = _conditions(batch, time, model_dim)
    return block(
        mano,
        camera,
        mano_pos,
        camera_pos,
        condition,
        condition_pos,
        hamer,
        vggt,
        bbox,
        mano_pos,
        vggt_pos,
        mano_pos,
        emb,
    )


def test_block_forward_shapes() -> None:
    block = _block()
    b, t = 2, 4
    mano = torch.randn(b, 2 * t, 32)
    camera = torch.randn(b, t, 32)
    emb = torch.randn(b, 32)

    m, c = _forward(block, mano, camera, emb)
    assert m.shape == mano.shape
    assert c.shape == camera.shape
    assert m.isfinite().all() and c.isfinite().all()


def test_block_near_identity_at_init() -> None:
    # zero-init AdaLN gates make FFN/attention contribute ~0, so output ~ input
    block = _block()
    b, t = 1, 3
    mano = torch.randn(b, 2 * t, 32)
    camera = torch.randn(b, t, 32)
    emb = torch.randn(b, 32)

    m, c = _forward(block, mano, camera, emb)
    # All attention and FFN paths are gated by zero-init gates, so the block is
    # near-identity at initialization.
    assert torch.allclose(m, mano, atol=1e-4)
    assert torch.allclose(c, camera, atol=1e-4)


def test_block_hybrid_attention_uses_latent_and_condition_kv() -> None:
    block = _block()
    b, t = 1, 3
    mano = torch.randn(b, 2 * t, 32)
    camera = torch.randn(b, t, 32)
    emb = torch.randn(b, 32)
    seen: dict[str, tuple[int, int, int]] = {}

    def record(name: str):
        def hook(_module, args):
            seen[name] = (args[0].shape[1], args[1].shape[1], args[2].shape[1])

        return hook

    block.cross_mano.register_forward_pre_hook(record("mano"))
    block.cross_camera.register_forward_pre_hook(record("camera"))
    _forward(block, mano, camera, emb)

    # Segment 0 uses own latent K/V plus [HaMeR, keypoint, bbox].
    assert seen["mano"] == (2 * t, 8 * t, 8 * t)
    assert seen["camera"] == (t, 22 * t, 22 * t)


def test_block_dual_attention_uses_stream_local_kv() -> None:
    block = _block()
    b, t = 1, 3
    mano = torch.randn(b, 2 * t, 32)
    camera = torch.randn(b, t, 32)
    emb = torch.randn(b, 32)
    seen: dict[str, tuple[int, int]] = {}

    def record(name: str):
        def hook(_module, args):
            seen[name] = (args[0].shape[1], args[1].shape[1])

        return hook

    block.dual_mano.register_forward_pre_hook(record("mano"))
    block.dual_camera.register_forward_pre_hook(record("camera"))
    _forward(block, mano, camera, emb)

    assert seen["mano"] == (2 * t, 8 * t)
    assert seen["camera"] == (t, 22 * t)


def test_block_finite_with_missing_condition_tokens() -> None:
    block = _block()
    b, t = 1, 2
    mano = torch.randn(b, 2 * t, 32)
    camera = torch.randn(b, t, 32)
    emb = torch.randn(b, 32)

    condition, condition_pos, hamer, vggt, bbox, mano_pos, camera_pos, vggt_pos = _conditions(
        b, t, 32
    )
    condition.zero_()
    hamer.zero_()
    vggt.zero_()
    bbox.zero_()
    m, c = block(
        mano,
        camera,
        mano_pos,
        camera_pos,
        condition,
        condition_pos,
        hamer,
        vggt,
        bbox,
        mano_pos,
        vggt_pos,
        mano_pos,
        emb,
    )
    assert m.isfinite().all() and c.isfinite().all()
