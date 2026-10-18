"""Training step: forward pass and loss computation.

The step computes all losses and returns the weighted total plus per-term
values. ``backward()`` and the optimizer step stay in the outer training loop.
Auxiliary losses (joints/acceleration/camera) can be gated by a warmup flag.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.loss.losses import (
    acceleration_loss,
    camera_flow_loss,
    camera_pose_loss,
    camera_trajectory_loss,
    flow_loss,
    joints3d_loss,
    joints3d_target_loss,
)
from src.model.flow.mmdit import MMDiT
from src.model.flow.rectified_flow import flow_interpolate
from src.model.flow.state import CameraFlowState, ConditionTokens, ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper

DEFAULT_WEIGHTS = {
    "flow": 1.0,
    "camera_flow": 1.0,
    "joints3d": 0.5,
    "acceleration": 0.02,
    "camera_pose": 0.1,
    "camera_trajectory": 0.1,
}


def train_step(
    *,
    model: MMDiT,
    mano_wrapper: ManoWrapper,
    target_mano: ManoFlowState | None,
    target_camera: CameraFlowState | None,
    conditions: ConditionTokens,
    target_joints3d: Tensor | None = None,
    joint_valid: Tensor | None = None,
    flow_time: Tensor,
    noise_mano: ManoFlowState,
    noise_camera: CameraFlowState,
    weights: dict[str, float] | None = None,
    auxiliary_enabled: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run one forward pass and return ``(total_loss, per_term_losses)``."""
    weights = dict(weights or DEFAULT_WEIGHTS)
    selected = {name for name, weight in weights.items() if weight > 0}
    if "flow" in selected and target_mano is None:
        raise ValueError("loss 'flow' requires target_mano")
    if {"camera_flow", "camera_pose", "camera_trajectory"} & selected and target_camera is None:
        raise ValueError("camera losses require target_camera")
    if "joints3d" in selected and target_joints3d is None and target_mano is None:
        raise ValueError("loss 'joints3d' requires target_joints3d or target_mano")
    if "joints3d" in selected and target_joints3d is not None and joint_valid is None:
        raise ValueError("external loss 'joints3d' requires joint_valid")
    if "acceleration" in selected and target_mano is None:
        raise ValueError("loss 'acceleration' requires target_mano")

    x_t_mano = (
        flow_interpolate(noise_mano, target_mano, flow_time)
        if target_mano is not None and "flow" in selected
        else noise_mano
    )
    x_t_camera = (
        flow_interpolate(noise_camera, target_camera, flow_time)
        if target_camera is not None and ({"camera_flow", "camera_pose", "camera_trajectory"} & selected)
        else noise_camera
    )

    prediction_mano, prediction_camera = model(
        x_t_mano, x_t_camera, flow_time, conditions
    )

    terms: dict[str, Tensor] = {}
    if "flow" in selected:
        terms["flow"] = flow_loss(prediction_mano, target_mano)
    if "camera_flow" in selected:
        terms["camera_flow"] = camera_flow_loss(prediction_camera, target_camera)
    if auxiliary_enabled and "joints3d" in selected:
        terms["joints3d"] = (
            joints3d_target_loss(prediction_mano, target_joints3d, joint_valid, mano_wrapper)
            if target_joints3d is not None
            else joints3d_loss(prediction_mano, target_mano, mano_wrapper)
        )
    if auxiliary_enabled and "acceleration" in selected:
        terms["acceleration"] = acceleration_loss(
            prediction_mano, target_mano, mano_wrapper
        )
    if auxiliary_enabled and "camera_pose" in selected:
        terms["camera_pose"] = camera_pose_loss(prediction_camera, target_camera)
    if auxiliary_enabled and "camera_trajectory" in selected:
        terms["camera_trajectory"] = camera_trajectory_loss(
            prediction_camera, target_camera
        )

    total = torch.zeros((), device=prediction_mano.translation.device)
    for name, value in terms.items():
        if weights.get(name, 0.0) > 0:
            total = total + weights[name] * value

    return total, terms
