"""Generic direct-joint flow objective and training helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from src.model.flow.joint import JointFlowModel, JointFlowState


def interpolate(noise: JointFlowState, target: JointFlowState, time: Tensor) -> JointFlowState:
    factor = time.reshape((-1,) + (1,) * (noise.joints.ndim - 1))
    return JointFlowState((1 - factor) * noise.joints + factor * target.joints)


def loss(prediction: JointFlowState, target: JointFlowState, valid: Tensor) -> Tensor:
    if prediction.joints.shape != target.joints.shape or valid.shape != target.joints.shape[:-1]:
        raise ValueError("joint flow target and validity shapes do not match")
    mask = valid.to(device=prediction.joints.device, dtype=torch.bool)
    target_value = target.joints.to(device=prediction.joints.device, dtype=prediction.joints.dtype)
    mask = mask & torch.isfinite(target_value).all(dim=-1)
    if not mask.any():
        raise ValueError("joint flow target has no valid points")
    return (prediction.joints - target_value).abs().masked_select(mask.unsqueeze(-1)).mean()


def train_joint_step(
    model: JointFlowModel,
    target: JointFlowState,
    valid: Tensor,
    time: Tensor,
    noise: JointFlowState,
) -> tuple[Tensor, JointFlowState]:
    prediction = model(interpolate(noise, target, time), time)
    return loss(prediction, target, valid), prediction
