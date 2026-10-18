"""Compute the complete metric set from saved prediction/target tensors."""

from __future__ import annotations

from torch import Tensor

from .aligned import pa_mpjpe_mm, wa_mpjpe_mm, wa_mpjpe_scaled_mm
from .camera import camera_metrics
from .motion import acceleration_error_mmps2, endpoint_drift_mm, velocity_error_mmps
from .world import world_metrics


def compute_metrics(
    prediction_mano,
    target_mano,
    prediction_camera,
    target_camera,
    prediction_joints: Tensor,
    target_joints: Tensor,
) -> dict[str, Tensor]:
    """Compute all hand/camera/motion metrics for one saved window."""
    values = world_metrics(
        prediction_joints,
        target_joints,
        prediction_mano.root_rotation,
        target_mano.root_rotation,
        prediction_mano.local_rotation,
        target_mano.local_rotation,
    )
    values.update(
        {
            "pa_mpjpe_mm": pa_mpjpe_mm(prediction_joints, target_joints),
            "wa_mpjpe_mm": wa_mpjpe_mm(prediction_joints, target_joints),
            "wa_mpjpe_scaled_mm": wa_mpjpe_scaled_mm(
                prediction_joints, target_joints
            ),
            "velocity_error_mmps": velocity_error_mmps(
                prediction_joints, target_joints
            ),
            "acceleration_error_mmps2": acceleration_error_mmps2(
                prediction_joints, target_joints
            ),
            "endpoint_drift_mm": endpoint_drift_mm(
                prediction_joints, target_joints
            ),
        }
    )
    values.update(
        camera_metrics(
            prediction_camera.rotation,
            target_camera.rotation,
            prediction_camera.translation,
            target_camera.translation,
        )
    )
    return values
