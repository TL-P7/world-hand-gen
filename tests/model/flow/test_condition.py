from __future__ import annotations

import torch

from src.model.flow.condition import ConditionEncoder


def _encoder() -> ConditionEncoder:
    return ConditionEncoder(hand_dim=1024, camera_dim=2048, model_dim=512)


def _inputs(b: int, t: int, dim: int = 512):
    hamer = torch.randn(b, 2, t, 1024)
    vggt = torch.randn(b, t, 17, 2048)
    bbox = torch.randn(b, 2, t, 4)
    keypoints = torch.rand(b, 2, t, 21, 3)
    masks = (
        torch.ones(b, 2, t, dtype=torch.bool),
        torch.ones(b, t, dtype=torch.bool),
        torch.ones(b, 2, t, dtype=torch.bool),
    )
    return hamer, vggt, bbox, keypoints, masks


def test_condition_encoder_shapes() -> None:
    encoder = _encoder()
    b, t = 2, 4
    hamer, vggt, bbox, keypoints, masks = _inputs(b, t)

    out = encoder(hamer, vggt, bbox, *masks, keypoints)

    assert out.hamer.shape == (b, 2 * t, 512)
    assert out.vggt.shape == (b, t * 17, 512)
    assert out.bbox.shape == (b, 2 * t, 512)
    assert out.keypoints.shape == (b, 2 * t, 512)


def test_condition_encoder_uses_independent_missing_tokens() -> None:
    encoder = _encoder()
    b, t = 2, 3
    hamer, vggt, bbox, keypoints, masks = _inputs(b, t)
    hamer_mask, vggt_mask, bbox_mask = masks
    hamer_mask[0, 0, 0] = False
    keypoints[1, 1, 2, :, 2] = 0

    out = encoder(hamer, vggt, bbox, hamer_mask, vggt_mask, bbox_mask, keypoints)

    assert torch.allclose(out.hamer[0, 0], encoder.hamer_missing)
    assert torch.allclose(out.keypoints[1, t + 2], encoder.keypoint_missing)
    assert not torch.allclose(out.keypoints[0, 0], encoder.keypoint_missing)


def test_condition_encoder_bbox_and_keypoint_are_trainable() -> None:
    encoder = _encoder()
    hamer, vggt, bbox, keypoints, masks = _inputs(1, 2)
    out = encoder(hamer, vggt, bbox, *masks, keypoints)
    out.bbox.sum().backward()
    assert encoder.bbox[0].weight.grad is not None
