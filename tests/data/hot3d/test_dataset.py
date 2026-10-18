from __future__ import annotations

import numpy as np
import pytest

from src.data import hot3d
from src.data.hot3d.dataset import Hot3DDataset


@pytest.fixture(scope="module")
def dataset():
    pytest.importorskip("projectaria_tools")
    clip = hot3d.clip_ids()[0]
    frame = hot3d.frame_ids(clip)[0]
    rectifier = hot3d.Hot3DRectifier(output_size=384)
    return Hot3DDataset(clip, [frame], rectifier)


def test_dataset_is_lazy(dataset) -> None:
    # Nothing is read or rectified until an accessor is called.
    assert dataset._images == {}
    assert dataset._boxes == {}
    assert dataset._calibrations == {}


def test_image_rectified_and_cached(dataset) -> None:
    frame = dataset.frames[0]
    image = dataset.image(frame)
    assert image.shape == (384, 384, 3)
    assert image.dtype == np.uint8
    # Second call hits the in-memory cache and returns the same array.
    assert dataset.image(frame) is image


def test_hand_box_rectified_and_cached(dataset) -> None:
    frame = dataset.frames[0]
    box = dataset.hand_box("left", frame)
    assert box is not None
    assert box.shape == (4,)
    assert box[2] > box[0] and box[3] > box[1]
    assert dataset.hand_box("left", frame) is box


def test_image_cache_skips_decode_and_calibration(dataset, tmp_path) -> None:
    frame = dataset.frames[0]
    cached = Hot3DDataset(
        dataset.clip_id, [frame], dataset.rectifier, root=hot3d.DEFAULT_ROOT,
        image_cache_root=tmp_path,
    )
    expected = cached.image(frame)
    hit = Hot3DDataset(
        dataset.clip_id, [frame], dataset.rectifier, root=hot3d.DEFAULT_ROOT,
        image_cache_root=tmp_path,
    )

    def fail(_frame):
        raise AssertionError("cache hit attempted calibration")

    hit._calibration = fail
    actual = hit.image(frame)
    assert np.array_equal(actual, expected)
    assert hit.image(frame) is actual


def test_box_cache_preserves_missing_value(dataset, tmp_path) -> None:
    frame = dataset.frames[0]
    cached = Hot3DDataset(
        dataset.clip_id, [frame], dataset.rectifier, root=hot3d.DEFAULT_ROOT,
        image_cache_root=tmp_path,
    )
    cached._raw_hands = lambda _frame: {"left": {}, "right": {}}
    assert cached.hand_box("left", frame) is None
    hit = Hot3DDataset(
        dataset.clip_id, [frame], dataset.rectifier, root=hot3d.DEFAULT_ROOT,
        image_cache_root=tmp_path,
    )
    hit._raw_hands = lambda _frame: (_ for _ in ()).throw(AssertionError("cache miss"))
    assert hit.hand_box("left", frame) is None
