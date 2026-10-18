from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.data.hot3d.cache import HaMeRTokenCache, _make_hamer_crop
from src.model.flow.condition import ConditionEncoder


class _FakeModel:
    IMAGE_SIZE = 256
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    @staticmethod
    def get(name, default=None):
        return default


class _FakeCfg:
    MODEL = _FakeModel()


class _FakeHamer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.model_cfg = _FakeCfg()
        self.batch_sizes = []

    def forward(self, crops):
        self.batch_sizes.append(crops.shape[0])
        return torch.arange(crops.shape[0], dtype=torch.float32).unsqueeze(1).expand(-1, 1024)


class _FakeDataset:
    def __init__(self, frames=(0, 1, 2)):
        self.clip_id = "001849"
        self.frames = list(frames)
        self.image_calls = []
        self.hidden = set()

    def image(self, frame):
        self.image_calls.append(frame)
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _box_inputs(dataset):
    boxes = torch.zeros(2, len(dataset.frames), 4)
    valid = torch.ones(2, len(dataset.frames), dtype=torch.bool)
    for hi, hand in enumerate(("left", "right")):
        for ti, frame in enumerate(dataset.frames):
            if (hand, frame) in dataset.hidden:
                valid[hi, ti] = False
            else:
                boxes[hi, ti] = torch.tensor([1.0, 1.0, 5.0, 5.0])
    return boxes, valid


def test_path_and_save_load_roundtrip(tmp_path):
    cache = HaMeRTokenCache(cache_root=tmp_path, hamer=_FakeHamer())
    path = cache._path("001849", 7, "left")
    assert path == tmp_path / "hamer" / "001849" / "000007_left.pt"
    cache._save(path, torch.full((1024,), 3.0))
    torch.testing.assert_close(cache._load(path), torch.full((1024,), 3.0))


@pytest.mark.parametrize("side", [0, 1])
def test_boundary_crop_is_black_before_normalization(side):
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    # The visible bbox is clipped, but HaMeR expands its crop beyond the image.
    box = np.array([0, 0, 5, 5], dtype=np.float32)
    crop = _make_hamer_crop(_FakeCfg(), image, box, side)
    pixels = crop * torch.tensor(_FakeModel.IMAGE_STD)[:, None, None]
    pixels += torch.tensor(_FakeModel.IMAGE_MEAN)[:, None, None]
    assert crop.shape == (3, 256, 256)
    torch.testing.assert_close(pixels[:, 0, 0], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(pixels[:, 128, 128], torch.ones(3), atol=1e-6, rtol=0)


def test_cache_batches_only_new_visible_crops(tmp_path):
    hamer = _FakeHamer()
    cache = HaMeRTokenCache(cache_root=tmp_path, hamer=hamer)
    first_dataset = _FakeDataset(frames=(0,))
    first, _ = cache.get_tokens(first_dataset, *_box_inputs(first_dataset))
    dataset = _FakeDataset()
    tokens, valid = cache.get_tokens(dataset, *_box_inputs(dataset))
    assert hamer.batch_sizes == [2, 4]
    assert valid.all()
    torch.testing.assert_close(first[:, 0], tokens[:, 0])
    dataset.image_calls.clear()
    repeated, repeated_valid = cache.get_tokens(dataset, *_box_inputs(dataset))
    assert hamer.batch_sizes == [2, 4] and dataset.image_calls == []
    assert torch.equal(tokens, repeated) and torch.equal(valid, repeated_valid)


def test_invisible_hand_ignores_stale_file_and_uses_learned_missing(tmp_path):
    hamer = _FakeHamer()
    cache = HaMeRTokenCache(cache_root=tmp_path, hamer=hamer)
    dataset = _FakeDataset(frames=(0, 1))
    dataset.hidden.add(("left", 1))
    cache._save(cache._path(dataset.clip_id, 1, "left"), torch.full((1024,), 9.0))
    tokens, valid = cache.get_tokens(dataset, *_box_inputs(dataset))
    assert valid.tolist() == [[True, False], [True, True]]
    assert hamer.batch_sizes == [3]
    assert not tokens[0, 1].any()

    encoder = ConditionEncoder(model_dim=16)
    with torch.no_grad():
        encoder.hamer_missing.fill_(2.0)
        encoder.bbox_missing.fill_(3.0)
    condition = encoder(
        tokens[None], torch.zeros(1, 2, 17, 2048), torch.zeros(1, 2, 2, 4),
        valid[None], torch.ones(1, 2, dtype=torch.bool), valid[None],
    )
    torch.testing.assert_close(condition.hamer[0, 1], encoder.hamer_missing)
    torch.testing.assert_close(condition.bbox[0, 1], encoder.bbox_missing)
    (condition.hamer[0, 1].sum() + condition.bbox[0, 1].sum()).backward()
    torch.testing.assert_close(encoder.hamer_missing.grad, torch.ones(16))
    torch.testing.assert_close(encoder.bbox_missing.grad, torch.ones(16))


def test_all_invisible_never_reads_image_or_runs_hamer(tmp_path):
    hamer = _FakeHamer()
    cache = HaMeRTokenCache(cache_root=tmp_path, hamer=hamer)
    dataset = _FakeDataset(frames=(0,))
    dataset.hidden = {("left", 0), ("right", 0)}
    for hand in ("left", "right"):
        cache._save(cache._path(dataset.clip_id, 0, hand), torch.ones(1024))
    tokens, valid = cache.get_tokens(dataset, *_box_inputs(dataset))
    assert hamer.batch_sizes == [] and dataset.image_calls == []
    assert not valid.any() and not tokens.any()
