from __future__ import annotations

import torch

from src.eval.metric.aligned import (
    pa_mpjpe_mm,
    wa_mpjpe_mm,
    wa_mpjpe_scaled_mm,
)
from src.eval.metric.camera import camera_metrics
from src.eval.metric.compute import compute_metrics
from src.eval.metric.motion import (
    acceleration_error_mmps2,
    endpoint_drift_mm,
    velocity_error_mmps,
)
from src.eval.metric.world import world_metrics
from src.model.flow.state import CameraFlowState, ManoFlowState

IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _identity_mano(b, t, transl_offset=0.0):
    translation = torch.zeros(b, 2, t, 3)
    translation[..., 0] = transl_offset
    root = IDENTITY_6D.expand(b, 2, t, 6).clone()
    local = IDENTITY_6D.expand(b, 2, t, 15, 6).clone()
    joints = torch.zeros(b, 2, t, 16, 3)
    joints[..., 0] = transl_offset
    return joints, translation, root, local


def test_world_metrics_identical_is_zero() -> None:
    j, t, r, l = _identity_mano(1, 4)
    metrics = world_metrics(j, j, r, r, l, l)
    for name, value in metrics.items():
        assert torch.allclose(value, torch.tensor(0.0), atol=1e-5), name


def test_world_mpjpe_detects_translation_offset() -> None:
    # prediction shifted by 100mm relative to target.
    jp, tp, rp, lp = _identity_mano(1, 4, transl_offset=0.1)
    jt, tt, rt, lt = _identity_mano(1, 4, transl_offset=0.0)
    metrics = world_metrics(jp, jt, rp, rt, lp, lt)
    # After first-frame align, prediction at t0 subtracts its own offset, so
    # constant offset cancels -> mpjpe should be ~0.
    assert metrics["mpjpe_mm"] < 1e-3


def test_world_w_alignment_uses_first_two_frames_only() -> None:
    # A constant offset is exactly fitted by the first two frames, while the
    # later frames contain a 100mm drift that must remain in W-MPJPE.
    target = torch.randn(1, 2, 4, 16, 3)
    prediction = target + torch.tensor([0.2, -0.1, 0.05])
    prediction[:, :, 2:] += torch.tensor([0.1, 0.0, 0.0])
    identity_r = IDENTITY_6D.expand(1, 2, 4, 6).clone()
    identity_l = IDENTITY_6D.expand(1, 2, 4, 15, 6).clone()
    metrics = world_metrics(
        prediction,
        target,
        identity_r,
        identity_r,
        identity_l,
        identity_l,
    )
    assert metrics["w_mpjpe_mm"] > 20.0
    assert metrics["w_mpjpe_scaled_mm"] > 20.0


def test_world_mpjpe_scaled_detects_scale() -> None:
    # prediction is target scaled 2x in motion (different offsets).
    jt, tt, rt, lt = _identity_mano(1, 4, transl_offset=0.0)
    # target has some motion; prediction is 2x that motion
    tt_motion = torch.zeros(1, 2, 4, 3)
    tt_motion[..., 0] = torch.arange(4).float() / 100.0
    jt_motion = tt_motion.unsqueeze(3).expand(1, 2, 4, 16, 3)
    tp_motion = 2 * tt_motion
    jp_motion = 2 * jt_motion
    metrics = world_metrics(jp_motion, jt_motion, rt, rt, lt, lt)
    # scaled error should be much smaller than unscaled.
    assert metrics["mpjpe_scaled_mm"] < metrics["mpjpe_mm"]


def test_world_metrics_use_decoded_wrist_root() -> None:
    batch, hands, time = 1, 2, 4
    prediction_joints = torch.zeros(batch, hands, time, 16, 3)
    target_joints = torch.zeros_like(prediction_joints)
    target_path = 0.3 + 0.1 * torch.arange(time, dtype=torch.float32)
    prediction_path = 0.8 + 0.2 * torch.arange(time, dtype=torch.float32)
    target_joints[..., 0] = target_path[None, None, :, None]
    prediction_joints[..., 0] = prediction_path[None, None, :, None]

    root = IDENTITY_6D.expand(batch, hands, time, 6).clone()
    local = IDENTITY_6D.expand(batch, hands, time, 15, 6).clone()
    state_translation = torch.zeros(batch, hands, time, 3)
    prediction_mano = ManoFlowState(state_translation, root, local)
    target_mano = ManoFlowState(state_translation.clone(), root.clone(), local.clone())
    camera_rotation = IDENTITY_6D.expand(batch, time, 6).clone()
    camera_translation = torch.zeros(batch, time, 3)
    prediction_camera = CameraFlowState(camera_rotation, camera_translation)
    target_camera = CameraFlowState(camera_rotation.clone(), camera_translation.clone())

    metrics = compute_metrics(
        prediction_mano,
        target_mano,
        prediction_camera,
        target_camera,
        prediction_joints,
        target_joints,
    )

    # The state translations are both zero, but the decoded wrists have a
    # 2x relative motion after their different first-frame offsets.
    assert torch.allclose(metrics["mpjpe_mm"], torch.tensor(150.0), atol=1e-5)
    assert torch.allclose(
        metrics["mpjpe_scaled_mm"], torch.tensor(0.0), atol=1e-4
    )
    assert torch.allclose(
        metrics["root_translation_error_mm"], torch.tensor(150.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_translation_error_scaled_mm"], torch.tensor(0.0), atol=1e-4
    )


def test_pa_mpjpe_zero_for_similarity_related() -> None:
    # prediction = 2x scale + rotation + translation of target, per-pose.
    target = torch.randn(2, 2, 3, 16, 3)
    # apply a fixed per-pose similarity: scale 2, no rotation, + translation
    prediction = 2.0 * target + torch.tensor([1.0, 2.0, 3.0])
    assert pa_mpjpe_mm(prediction, target) < 1e-3


def test_wa_mpjpe_rigid_and_scaled_global_similarity() -> None:
    target = torch.randn(1, 2, 3, 16, 3)
    prediction = 2.0 * target + torch.tensor([1.0, 2.0, 3.0])
    # A rigid fit cannot remove the scale change, but global similarity can.
    assert wa_mpjpe_mm(prediction, target) > 1.0
    assert wa_mpjpe_scaled_mm(prediction, target) < 1e-3


def test_velocity_and_acceleration_zero_for_identical() -> None:
    t = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1) / 1000.0
    joints = t.expand(1, 2, 5, 16, 3)
    assert velocity_error_mmps(joints, joints) == 0
    assert acceleration_error_mmps2(joints, joints) == 0


def test_endpoint_drift_detects_relative_motion() -> None:
    # target moves 100mm over the sequence; prediction moves 200mm.
    tgt = torch.zeros(1, 2, 4, 16, 3)
    tgt[:, :, -1, :, 0] = 0.1
    pred = torch.zeros(1, 2, 4, 16, 3)
    pred[:, :, -1, :, 0] = 0.2
    drift = endpoint_drift_mm(pred, tgt)
    assert torch.allclose(drift, torch.tensor(100.0), atol=1e-2)


def test_camera_metrics_identical_is_zero() -> None:
    rot = IDENTITY_6D.expand(1, 4, 6).clone()
    transl = torch.zeros(1, 4, 3)
    metrics = camera_metrics(rot, rot, transl, transl)
    for name, value in metrics.items():
        assert torch.allclose(value, torch.tensor(0.0), atol=1e-5), name
