"""Hand world-space metrics with first-frame and W alignment.

Positions are reported in millimetres and rotations in degrees. ``mpjpe``
retains first-frame translation alignment. W-MPJPE fits one transform from the
first two frames (all hands and joints) and applies it to the full sequence:
its unscaled variant uses Kabsch rotation plus translation, while its scaled
variant also fits one positive scale.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.geometry.rotation import geodesic_distance, rotation_6d_to_matrix

from .align import align_translation, fit_scale
from .aligned import fit_rigid, fit_similarity

_RAD2DEG = 180.0 / 3.141592653589793


def _first_frame(value: Tensor) -> Tensor:
    """Extract the first-frame slice, broadcast over the joint axis."""
    return value[:, :, :1]


def _fit_w_transform(
    prediction_joints: Tensor, target_joints: Tensor, scaled: bool
) -> Tensor:
    """Fit W alignment on frames 0 and 1, then transform every sequence point."""
    pred_fit = prediction_joints[:, :, :2].reshape(prediction_joints.shape[0], -1, 3)
    tgt_fit = target_joints[:, :, :2].reshape(target_joints.shape[0], -1, 3)
    pred_all = prediction_joints.reshape(prediction_joints.shape[0], -1, 3)
    if scaled:
        rotation, scale, translation = fit_similarity(pred_fit, tgt_fit)
        aligned = scale[:, None, None] * torch.matmul(
            pred_all, rotation.transpose(-2, -1)
        ) + translation[:, None, :]
    else:
        rotation, translation = fit_rigid(pred_fit, tgt_fit)
        aligned = torch.matmul(pred_all, rotation.transpose(-2, -1)) + translation[:, None, :]
    return aligned.reshape_as(prediction_joints)


def world_metrics(
    prediction_joints: Tensor,  # [B,2,T,16,3] meters, decoded MANO joints
    target_joints: Tensor,  # [B,2,T,16,3] meters, decoded MANO joints
    prediction_root: Tensor,  # [B,2,T,6] 6D
    target_root: Tensor,  # [B,2,T,6] 6D
    prediction_local: Tensor,  # [B,2,T,15,6] 6D
    target_local: Tensor,  # [B,2,T,15,6] 6D
) -> dict[str, Tensor]:
    pred_r0 = rotation_6d_to_matrix(prediction_root)
    tgt_r0 = rotation_6d_to_matrix(target_root)

    # The decoded MANO wrist is the root used by all position metrics.  The
    # flow state's ``translation`` is only the MANO ``transl`` input offset.
    pred_wrist = prediction_joints[..., 0, :]
    tgt_wrist = target_joints[..., 0, :]
    pred_t0 = _first_frame(pred_wrist)
    tgt_t0 = _first_frame(tgt_wrist)

    # MPJPE keeps its historical first-frame translation-only alignment.
    pred_joints_trans = align_translation(prediction_joints, pred_t0.unsqueeze(3))
    tgt_joints_trans = align_translation(target_joints, tgt_t0.unsqueeze(3))

    # W-MPJPE fits prediction and target jointly on frames 0 and 1, then applies
    # that single transform to every frame in the sequence.
    pred_joints_w = _fit_w_transform(prediction_joints, target_joints, scaled=False)
    pred_joints_w_scaled = _fit_w_transform(
        prediction_joints, target_joints, scaled=True
    )

    # Root translation (translation-only align) uses the decoded wrist root.
    pred_trans_align = align_translation(pred_wrist, pred_t0)
    tgt_trans_align = align_translation(tgt_wrist, tgt_t0)

    def mpjpe(pred: Tensor, tgt: Tensor) -> Tensor:
        return (pred - tgt).norm(dim=-1).mean() * 1000.0

    def scaled_mpjpe(pred: Tensor, tgt: Tensor) -> Tensor:
        scale = fit_scale(pred, tgt)
        return (scale * pred - tgt).norm(dim=-1).mean() * 1000.0

    # Root rotation error: translation-aligned and rigid-aligned.
    root_rot_trans = geodesic_distance(pred_r0, tgt_r0).mean() * _RAD2DEG
    pred_r0_aligned = _first_frame(pred_r0).transpose(-2, -1) @ pred_r0
    tgt_r0_aligned = _first_frame(tgt_r0).transpose(-2, -1) @ tgt_r0
    root_rot_rigid = geodesic_distance(pred_r0_aligned, tgt_r0_aligned).mean() * _RAD2DEG

    # Local rotation error: joint-local, invariant to global alignment.
    pred_local = rotation_6d_to_matrix(prediction_local)
    tgt_local = rotation_6d_to_matrix(target_local)
    local_rot = geodesic_distance(pred_local, tgt_local).mean() * _RAD2DEG

    return {
        "mpjpe_mm": mpjpe(pred_joints_trans, tgt_joints_trans),
        "mpjpe_scaled_mm": scaled_mpjpe(pred_joints_trans, tgt_joints_trans),
        "w_mpjpe_mm": mpjpe(pred_joints_w, target_joints),
        "w_mpjpe_scaled_mm": mpjpe(pred_joints_w_scaled, target_joints),
        "root_translation_error_mm": mpjpe(pred_trans_align, tgt_trans_align),
        "root_translation_error_scaled_mm": scaled_mpjpe(
            pred_trans_align, tgt_trans_align
        ),
        "root_rotation_error_deg": root_rot_trans,
        "w_root_rotation_error_deg": root_rot_rigid,
        "local_rotation_error_deg": local_rot,
        "w_local_rotation_error_deg": local_rot,
    }
