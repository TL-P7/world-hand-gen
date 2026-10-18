from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.data.hot3d.cache import VGGTOmegaTokenCache


class FakeVGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0))
        self.calls = 0

    @torch.inference_mode()
    def forward(self, images):
        self.calls += 1
        assert images.shape == (1, 64, 3, 8, 8)
        return images[:, :, 0, 0, 0, None, None].expand(1, 64, 17, 2048)


def test_cache_reuses_window_without_images_or_model(tmp_path):
    model = FakeVGGT()
    reads = []

    def image(frame):
        reads.append(frame)
        return np.full((8, 8, 3), frame, dtype=np.uint8)

    dataset = SimpleNamespace(clip_id="001849", frames=list(range(64)), image=image)
    cache = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=model)
    tokens = cache.get_tokens(dataset)
    assert reads == list(range(64)) and model.calls == 1
    assert tokens.shape == (1, 64, 17, 2048)
    assert tokens.device.type == "cpu" and not tokens.requires_grad
    assert not tokens.is_inference()
    torch.testing.assert_close(tokens[0, :, 0, 0], torch.arange(64) / 255)
    reads.clear()
    repeated = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=model).get_tokens(dataset)
    assert reads == [] and model.calls == 1
    assert torch.equal(tokens, repeated)


def test_cache_distinguishes_clips_and_both_windows(tmp_path):
    model = FakeVGGT()
    cache = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=model)
    image = lambda frame: np.full((8, 8, 3), frame, dtype=np.uint8)
    for clip, start in [("001849", 0), ("001849", 64), ("001850", 0)]:
        cache.get_tokens(SimpleNamespace(clip_id=clip, frames=list(range(start, start + 64)), image=image))
    assert model.calls == 3
    assert len(list(tmp_path.rglob("*.pt"))) == 3
    assert not list(tmp_path.rglob("*.tmp"))


def test_cache_only_hit_and_miss_do_not_load_images(tmp_path):
    dataset = SimpleNamespace(
        clip_id="001849", frames=list(range(64)),
        image=lambda frame: pytest.fail("cache-only evaluation must not read images"),
    )
    cache = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=None)
    with pytest.raises(RuntimeError, match="missing VGGT cache"):
        cache.get_tokens(dataset)
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.tmp"))
    path = cache._path(dataset.clip_id, dataset.frames)
    path.parent.mkdir(parents=True)
    expected = torch.zeros(1, 64, 17, 2048)
    torch.save({"tokens": expected}, path)
    torch.testing.assert_close(cache.get_tokens(dataset), expected)


def test_partial_cache_only_computes_missing_window(tmp_path):
    model = FakeVGGT()
    cache = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=model)
    reads = []

    def image(frame):
        reads.append(frame)
        return np.full((8, 8, 3), frame, dtype=np.uint8)

    first = SimpleNamespace(clip_id="001849", frames=list(range(64)), image=image)
    second = SimpleNamespace(clip_id="001849", frames=list(range(64, 128)), image=image)
    path = cache._path(first.clip_id, first.frames)
    path.parent.mkdir(parents=True)
    torch.save({"tokens": torch.zeros(1, 64, 17, 2048)}, path)
    cache.get_tokens(first)
    assert reads == [] and model.calls == 0
    cache.get_tokens(second)
    assert reads == second.frames and model.calls == 1


def test_random_or_reordered_windows_cannot_reuse_eval_cache(tmp_path):
    cache = VGGTOmegaTokenCache(cache_root=tmp_path, vggt=FakeVGGT())
    for frames in [list(range(1, 65)), list(reversed(range(64))), [0] * 64]:
        with pytest.raises(ValueError, match="only supports"):
            cache.get_tokens(SimpleNamespace(clip_id="001849", frames=frames))
