"""Camera pose and trajectory metrics.

Rotation reported in degrees, translation in millimetres. Inputs are the
canonical relative camera transforms in the first model-camera reference frame;
absolute pose is first-frame aligned (translation-only or rigid), and trajectory
uses consecutive increments, which are already free of global offset. Raw
``camera_extrinsics`` must be converted by the adaptor before reaching here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.rotation import geodesic_distance, rotation_6d_to_matrix

from .align import align_rigid, align_translation, fit_scale

_RAD2DEG = 180.0 / 3.141592653589793
_MM = 1000.0


def camera_metrics(
    prediction: Tensor,  # [B,T,6] rotation 6D
    target: Tensor,  # [B,T,6] rotation 6D
    prediction_translation: Tensor,  # [B,T,3] metres
    target_translation: Tensor,  # [B,T,3] metres
) -> dict[str, Tensor]:
    pred_r = rotation_6d_to_matrix(prediction)
    tgt_r = rotation_6d_to_matrix(target)

    pred_t0 = prediction_translation[:, :1]
    tgt_t0 = target_translation[:, :1]

    # First-frame aligned translation.
    pred_trans = align_translation(prediction_translation, pred_t0)
    tgt_trans = align_translation(target_translation, tgt_t0)
    pred_rigid = align_rigid(prediction_translation, pred_t0, pred_r[:, :1])
    tgt_rigid = align_rigid(target_translation, tgt_t0, tgt_r[:, :1])

    def translation_error(pred: Tensor, tgt: Tensor) -> Tensor:
        return (pred - tgt).norm(dim=-1).mean() * _MM

    def translation_error_scaled(pred: Tensor, tgt: Tensor) -> Tensor:
        scale = fit_scale(pred, tgt)
        return (scale * pred - tgt).norm(dim=-1).mean() * _MM

    # Rotation: translation-aligned (absolute) and rigid-aligned (first frame).
    rotation_trans = geodesic_distance(pred_r, tgt_r).mean() * _RAD2DEG
    pred_r_aligned = pred_r[:, :1].transpose(-2, -1) @ pred_r
    tgt_r_aligned = tgt_r[:, :1].transpose(-2, -1) @ tgt_r
    rotation_rigid = geodesic_distance(pred_r_aligned, tgt_r_aligned).mean() * _RAD2DEG

    # Trajectory: consecutive increments.
    pred_delta = pred_r[:, :-1].transpose(-2, -1) @ pred_r[:, 1:]
    tgt_delta = tgt_r[:, :-1].transpose(-2, -1) @ tgt_r[:, 1:]
    traj_rotation = geodesic_distance(pred_delta, tgt_delta).mean() * _RAD2DEG

    pred_disp = prediction_translation[:, 1:] - prediction_translation[:, :-1]
    tgt_disp = target_translation[:, 1:] - target_translation[:, :-1]
    traj_translation = (pred_disp - tgt_disp).norm(dim=-1).mean() * _MM
    scale = fit_scale(pred_disp, tgt_disp)
    traj_translation_scaled = (scale * pred_disp - tgt_disp).norm(dim=-1).mean() * _MM

    return {
        "camera_rotation_error_deg": rotation_trans,
        "w_camera_rotation_error_deg": rotation_rigid,
        "camera_translation_error_mm": translation_error(pred_trans, tgt_trans),
        "camera_translation_error_scaled_mm": translation_error_scaled(
            pred_trans, tgt_trans
        ),
        "w_camera_translation_error_mm": translation_error(pred_rigid, tgt_rigid),
        "w_camera_translation_error_scaled_mm": translation_error_scaled(
            pred_rigid, tgt_rigid
        ),
        "camera_trajectory_rotation_error_deg": traj_rotation,
        "camera_trajectory_translation_error_mm": traj_translation,
        "camera_trajectory_translation_error_scaled_mm": traj_translation_scaled,
    }
