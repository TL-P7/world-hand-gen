from __future__ import annotations

from src.data import hot3d


def test_clip_ids_nonempty() -> None:
    assert hot3d.clip_ids()


def test_load_frame_returns_raw_rgb() -> None:
    clip = hot3d.clip_ids()[0]
    frame = hot3d.frame_ids(clip)[0]
    sample = hot3d.load_frame(clip, frame)
    assert sample.image.ndim == 3
    assert sample.image.shape[2] == 3
    assert sample.image.dtype.name == "uint8"
    assert "left" in sample.hands and "right" in sample.hands
    assert "214-1" in sample.cameras


def test_hand_box_coordinates() -> None:
    clip = hot3d.clip_ids()[0]
    frame = hot3d.frame_ids(clip)[0]
    sample = hot3d.load_frame(clip, frame)
    for hand in ("left", "right"):
        box = sample.hands[hand]["boxes_amodal"]["214-1"]
        assert len(box) == 4
        assert box[2] > box[0] and box[3] > box[1]
