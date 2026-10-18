import json

import pytest
import torch

from src.eval import protocol


def _window(clip, window, left_offscreen, right_offscreen):
    identity = torch.tensor([1., 0., 0., 0., 1., 0.])
    frames = list(range(window * 64, window * 64 + 64))
    mano = {
        "translation": torch.zeros(1, 2, 64, 3),
        "root_rotation": identity.expand(1, 2, 64, 6).clone(),
        "local_rotation": identity.expand(1, 2, 64, 15, 6).clone(),
    }
    camera = {
        "rotation": identity.expand(1, 64, 6).clone(),
        "translation": torch.zeros(1, 64, 3),
    }
    joints = torch.zeros(1, 2, 64, 16, 3)
    return {
        "clip": clip, "window": window, "frame_ids": frames,
        "prediction_mano": mano, "target_mano": mano,
        "prediction_camera": camera, "target_camera": camera,
        "prediction_joints": joints, "target_joints": joints,
        "missing_mano": [],
        "offscreen": {"left": [index < left_offscreen for index in frames],
                      "right": [index < right_offscreen for index in frames]},
    }


def test_offscreen_threshold_uses_each_hand_and_keeps_camera(tmp_path, monkeypatch):
    calls = []

    def masks(_root, clip, frames, _size):
        calls.append((clip, frames[0]))
        return {"left": [clip == "002" and index < 32 for index in range(64)],
                "right": [clip == "001" and index < 31 for index in range(64)]}

    evaluate, hand, camera, _visibility = protocol._metric()
    monkeypatch.setattr(protocol, "_metric", lambda: (evaluate, hand, camera, masks))
    windows = [_window("001", 0, 0, 0), _window("002", 0, 0, 0)]
    source = tmp_path / "predictions.pt"
    torch.save({"checkpoint_step": 2000, "windows": windows}, source)
    result = protocol.evaluate_saved(source, tmp_path / "result", cache_root=tmp_path / "cache",
                                     data_root=tmp_path, expected_windows=2)
    assert (result["hand_windows"], result["excluded_windows"]) == (1, 1)
    assert result["valid_windows_per_metric"]["mpjpe_mm"] == 1
    assert result["valid_windows_per_metric"]["camera_rotation_error_deg"] == 2
    assert len(calls) == 2
    assert len(json.loads((tmp_path / "result/visibility_masks.json").read_text())["windows"]) == 2
    with pytest.raises(FileExistsError):
        protocol.evaluate_saved(source, tmp_path / "result", cache_root=tmp_path / "cache", data_root=tmp_path)
    protocol.evaluate_saved(source, tmp_path / "second", cache_root=tmp_path / "cache", data_root=tmp_path)
    assert len(calls) == 2


def test_missing_mano_has_null_hand_metrics_but_camera_remains(tmp_path, monkeypatch):
    evaluate, hand, camera, _visibility = protocol._metric()
    monkeypatch.setattr(protocol, "_metric", lambda: (evaluate, hand, camera,
                       lambda *_args: {"left": [False] * 64, "right": [False] * 64}))
    window = _window("001", 0, 0, 0)
    window["missing_mano"] = ["left@0"]
    source = tmp_path / "predictions.pt"
    torch.save({"checkpoint_step": 4000, "windows": [window]}, source)
    summary = protocol.evaluate_saved(source, tmp_path / "result", cache_root=tmp_path / "cache",
                                      data_root=tmp_path, expected_windows=1)
    assert summary["metrics"]["mpjpe_mm"] is None
    assert summary["metrics"]["camera_rotation_error_deg"] == 0.0
