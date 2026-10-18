from pathlib import Path

import pytest
import torch

from src.eval.recompute import recompute


def _state_shapes():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    mano = {
        "translation": torch.zeros(1, 2, 4, 3),
        "root_rotation": identity.expand(1, 2, 4, 6).clone(),
        "local_rotation": identity.expand(1, 2, 4, 15, 6).clone(),
    }
    camera = {
        "rotation": identity.expand(1, 4, 6).clone(),
        "translation": torch.zeros(1, 4, 3),
    }
    joints = torch.zeros(1, 2, 4, 16, 3)
    return mano, camera, joints


def test_recompute_is_model_free_and_matches_zero_artifact():
    mano, camera, joints = _state_shapes()
    artifacts = [{
        "clip": "001849", "window": 0, "frame_ids": list(range(4)),
        "prediction_mano": mano, "target_mano": mano,
        "prediction_camera": camera, "target_camera": camera,
        "prediction_joints": joints, "target_joints": joints,
        "missing_mano": [],
    }]
    records, summary = recompute(artifacts)
    assert len(records) == 1
    assert summary["metrics"]["w_mpjpe_mm"] == 0.0
    assert summary["metrics"]["w_mpjpe_scaled_mm"] == 0.0
    assert summary["metrics"]["wa_mpjpe_mm"] == 0.0
    assert summary["metrics"]["wa_mpjpe_scaled_mm"] == 0.0


def test_recompute_uses_decoded_wrist_root():
    mano, camera, joints = _state_shapes()
    prediction_joints = joints.clone()
    target_joints = joints.clone()
    time = prediction_joints.shape[2]
    target_path = 0.3 + 0.1 * torch.arange(time, dtype=torch.float32)
    prediction_path = 0.8 + 0.2 * torch.arange(time, dtype=torch.float32)
    target_joints[..., 0] = target_path[None, None, :, None]
    prediction_joints[..., 0] = prediction_path[None, None, :, None]
    artifacts = [{
        "clip": "001849", "window": 0, "frame_ids": list(range(time)),
        "prediction_mano": mano, "target_mano": mano,
        "prediction_camera": camera, "target_camera": camera,
        "prediction_joints": prediction_joints, "target_joints": target_joints,
        "missing_mano": [],
    }]

    records, summary = recompute(artifacts)

    assert records[0]["mpjpe_mm"] == pytest.approx(150.0)
    assert records[0]["mpjpe_scaled_mm"] == pytest.approx(0.0, abs=1e-4)
    assert records[0]["root_translation_error_mm"] == pytest.approx(150.0)
    assert records[0]["root_translation_error_scaled_mm"] == pytest.approx(0.0, abs=1e-4)
