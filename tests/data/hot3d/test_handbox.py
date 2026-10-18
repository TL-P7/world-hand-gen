from __future__ import annotations

import numpy as np
import torch

from src.data.hot3d.handbox import HandBoxCache


class _Dataset:
    clip_id = "000001"
    frames = [3, 4, 5]

    def __init__(self):
        self.image_calls = []

    def image(self, frame):
        self.image_calls.append(frame)
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _Detector:
    def __init__(self):
        self.calls = []

    def __call__(self, images):
        self.calls.append(tuple(images.shape))
        output = []
        for index in range(images.shape[0]):
            boxes = torch.zeros(2, 4)
            valid = torch.tensor([True, index == 0])
            boxes[0] = torch.tensor([1.0, 1.0, 5.0, 5.0])
            if valid[1]:
                boxes[1] = torch.tensor([2.0, 2.0, 6.0, 6.0])
            output.append({"boxes": boxes, "valid": valid, "metadata": {"index": index}})
        return output


class _KeypointDetector(_Detector):
    def __call__(self, images):
        self.calls.append(tuple(images.shape))
        outputs = []
        for index in range(images.shape[0]):
            keypoints = torch.zeros(2, 21, 3)
            keypoints[:, :, 0] = 10.0 + index
            keypoints[:, :, 1] = 12.0
            keypoints[:, :, 2] = 0.9
            outputs.append({
                "boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0], [2.0, 3.0, 6.0, 7.0]]),
                "box_valid": torch.tensor([True, True]),
                "keypoints": keypoints,
            })
        return outputs


def test_handbox_detection_cache_keeps_keypoints_independent(tmp_path):
    dataset = _Dataset()
    detector = _KeypointDetector()
    cache = HandBoxCache(cache_root=tmp_path, detector=detector, batch_size=2)

    keypoints, boxes, valid, records = cache.get_detections(dataset)
    assert keypoints.shape == (2, 3, 21, 3)
    assert boxes.shape == (2, 3, 4)
    assert valid.all()
    assert "hamer_valid" not in records[0]
    detector.calls.clear()
    keypoints2, boxes2, valid2, _ = cache.get_detections(dataset)
    assert detector.calls == []
    torch.testing.assert_close(keypoints, keypoints2)
    torch.testing.assert_close(boxes, boxes2)
    assert torch.equal(valid, valid2)


def test_handbox_old_box_record_is_detection_cache_miss(tmp_path):
    dataset = _Dataset()
    detector = _KeypointDetector()
    cache = HandBoxCache(cache_root=tmp_path, detector=detector)
    path = cache._path(dataset.clip_id, dataset.frames[0])
    path.parent.mkdir(parents=True)
    torch.save({"boxes": torch.zeros(2, 4), "valid": torch.zeros(2, dtype=torch.bool)}, path)

    keypoints, _, _, _ = cache.get_detections(dataset)
    assert detector.calls
    assert keypoints.shape == (2, 3, 21, 3)


def test_handbox_batches_misses_and_hits_cache(tmp_path):
    dataset = _Dataset()
    detector = _Detector()
    cache = HandBoxCache(cache_root=tmp_path, detector=detector, batch_size=2)

    boxes, valid, records = cache.get_boxes(dataset)
    assert detector.calls == [(2, 3, 8, 8), (1, 3, 8, 8)]
    assert dataset.image_calls == [3, 4, 5]
    assert boxes.shape == (2, 3, 4)
    assert valid.tolist() == [[True, True, True], [True, False, True]]
    assert len(records) == 3

    detector.calls.clear()
    dataset.image_calls.clear()
    boxes2, valid2, _ = cache.get_boxes(dataset)
    assert detector.calls == []
    assert dataset.image_calls == []
    torch.testing.assert_close(boxes, boxes2)
    assert torch.equal(valid, valid2)


def test_handbox_does_not_fallback_for_invalid_hand(tmp_path):
    dataset = _Dataset()
    detector = _Detector()
    cache = HandBoxCache(cache_root=tmp_path, detector=detector)
    boxes, valid, _ = cache.get_boxes(dataset)
    assert not valid[1, 1]
    assert not boxes[1, 1].any()
