"""Aligned hand metrics: PA-MPJPE and global WA-MPJPE variants.

The transforms map prediction points to target points and reject reflections.
Rigid metrics fit only rotation and translation; similarity metrics additionally
fit one positive scale. All reported position errors are in millimetres.
"""

from __future__ import annotations

import torch
from torch import Tensor


def fit_rigid(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Fit a proper Kabsch transform mapping prediction points to target points.

    ``prediction`` and ``target`` have shape ``[...,N,3]``. The returned
    ``rotation`` uses row-vector convention, so aligned points are
    ``prediction @ rotation.T + translation``.
    """
    pred_center = prediction.mean(dim=-2)
    tgt_center = target.mean(dim=-2)
    pred_c = prediction - pred_center.unsqueeze(-2)
    tgt_c = target - tgt_center.unsqueeze(-2)

    covariance = pred_c.transpose(-2, -1) @ tgt_c
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(-2, -1) @ u.transpose(-2, -1)

    # Kabsch must use a proper rotation rather than a reflection.
    det = torch.linalg.det(rotation)
    correction = torch.eye(
        3, device=prediction.device, dtype=prediction.dtype
    ).expand(*rotation.shape[:-2], 3, 3).clone()
    correction[..., -1, -1] = torch.where(
        det < 0, torch.tensor(-1.0, device=prediction.device, dtype=prediction.dtype),
        torch.tensor(1.0, device=prediction.device, dtype=prediction.dtype),
    )
    rotation = vh.transpose(-2, -1) @ correction @ u.transpose(-2, -1)
    translation = tgt_center - torch.matmul(pred_center.unsqueeze(-2), rotation.transpose(-2, -1)).squeeze(-2)
    return rotation, translation


def fit_similarity(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a proper rotation, positive scale, and translation to target.

    Inputs have shape ``[...,N,3]``. Returns ``(rotation, scale, translation)``
    using the same row-vector convention as :func:`fit_rigid`.
    """
    pred_center = prediction.mean(dim=-2)
    tgt_center = target.mean(dim=-2)
    pred_c = prediction - pred_center.unsqueeze(-2)
    tgt_c = target - tgt_center.unsqueeze(-2)

    covariance = pred_c.transpose(-2, -1) @ tgt_c
    u, singular_values, vh = torch.linalg.svd(covariance)
    raw_rotation = vh.transpose(-2, -1) @ u.transpose(-2, -1)
    det = torch.linalg.det(raw_rotation)
    correction = torch.eye(
        3, device=prediction.device, dtype=prediction.dtype
    ).expand(*raw_rotation.shape[:-2], 3, 3).clone()
    correction[..., -1, -1] = torch.where(
        det < 0, torch.tensor(-1.0, device=prediction.device, dtype=prediction.dtype),
        torch.tensor(1.0, device=prediction.device, dtype=prediction.dtype),
    )
    rotation = vh.transpose(-2, -1) @ correction @ u.transpose(-2, -1)
    numerator = (singular_values * correction.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
    denominator = (pred_c * pred_c).sum(dim=(-2, -1)).clamp_min(1e-12)
    scale = (numerator / denominator).clamp_min(0.0)
    translation = tgt_center - scale.unsqueeze(-1) * torch.matmul(
        pred_center.unsqueeze(-2), rotation.transpose(-2, -1)
    ).squeeze(-2)
    return rotation, scale, translation


def fit_similarity_transform(
    prediction: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backward-compatible centered form of :func:`fit_similarity`."""
    rotation, scale, translation = fit_similarity(prediction, target)
    pred_center = prediction.mean(dim=-2, keepdim=True)
    tgt_center = target.mean(dim=-2, keepdim=True)
    del translation
    return rotation, scale, pred_center, tgt_center


def procrustes_align(prediction: Tensor, target: Tensor) -> Tensor:
    """Align each pose ``[...,J,3]`` to its target and return aligned points."""
    batch_shape = prediction.shape[:-2]
    pred_flat = prediction.reshape(-1, prediction.shape[-2], 3)
    tgt_flat = target.reshape(-1, target.shape[-2], 3)
    rotation, scale, translation = fit_similarity(pred_flat, tgt_flat)
    aligned = scale[:, None, None] * torch.einsum(
        "bnc,b dc->bnd", pred_flat, rotation
    ) + translation[:, None, :]
    return aligned.reshape(*batch_shape, prediction.shape[-2], 3)


def pa_mpjpe_mm(prediction: Tensor, target: Tensor) -> Tensor:
    """Per-pose Procrustes-aligned MPJPE in millimetres."""
    aligned = procrustes_align(prediction, target)
    return (aligned - target).norm(dim=-1).mean() * 1000.0


def _global_aligned(
    prediction: Tensor, target: Tensor, similarity: bool
) -> Tensor:
    """Apply one fitted transform per batch item to all sequence points."""
    pred_flat = prediction.reshape(prediction.shape[0], -1, 3)
    tgt_flat = target.reshape(target.shape[0], -1, 3)
    if similarity:
        rotation, scale, translation = fit_similarity(pred_flat, tgt_flat)
        aligned = scale[:, None, None] * torch.einsum(
            "bnc,bdc->bnd", pred_flat, rotation
        ) + translation[:, None, :]
    else:
        rotation, translation = fit_rigid(pred_flat, tgt_flat)
        aligned = torch.einsum("bnc,bdc->bnd", pred_flat, rotation) + translation[:, None, :]
    return aligned.reshape_as(prediction)


def wa_mpjpe_mm(prediction: Tensor, target: Tensor) -> Tensor:
    """Global rigid-aligned MPJPE in millimetres.

    One Kabsch transform is fitted per batch item over all hands, frames, and
    joints. No scale is fitted.
    """
    aligned = _global_aligned(prediction, target, similarity=False)
    return (aligned - target).norm(dim=-1).mean() * 1000.0


def wa_mpjpe_scaled_mm(prediction: Tensor, target: Tensor) -> Tensor:
    """Global similarity-aligned MPJPE in millimetres."""
    aligned = _global_aligned(prediction, target, similarity=True)
    return (aligned - target).norm(dim=-1).mean() * 1000.0
