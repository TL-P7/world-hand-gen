"""Direct loss functions for MANO and camera flow matching."""

from __future__ import annotations

import torch
from torch import Tensor

from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper

from src.geometry.rotation import geodesic_distance, rotation_6d_to_matrix

def flow_loss(prediction: ManoFlowState, target: ManoFlowState) -> Tensor:
    """Mean x0 MSE over all MANO flow fields."""
    values = (
        (prediction.translation - target.translation).square(),
        (prediction.root_rotation - target.root_rotation).square(),
        (prediction.local_rotation - target.local_rotation).square(),
    )
    return torch.cat(tuple(value.reshape(-1) for value in values)).mean()


def camera_flow_loss(prediction: CameraFlowState, target: CameraFlowState) -> Tensor:
    """Mean x0 MSE over camera rotation-6D and translation fields."""
    values = (
        (prediction.rotation - target.rotation).square(),
        (prediction.translation - target.translation).square(),
    )
    return torch.cat(tuple(value.reshape(-1) for value in values)).mean()


_PUBLIC_FROM_MANO = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)


def _public_joints(joints: Tensor) -> Tensor:
    """Reorder native MANO-16 joints into the public project order."""
    return joints[..., list(_PUBLIC_FROM_MANO), :]


def joints3d_loss(
    prediction: ManoFlowState, target: ManoFlowState, mano: ManoWrapper
) -> Tensor:
    """Mean L1 error between MANO joints decoded from both flow states."""
    return (mano(prediction) - mano(target)).abs().mean()


def joints3d_target_loss(
    prediction: ManoFlowState,
    target_joints: Tensor,
    valid: Tensor,
    mano: ManoWrapper,
) -> Tensor:
    """Masked L1 loss against an external public-order 3D joint target."""
    predicted = _public_joints(mano(prediction))
    if target_joints.shape != predicted.shape or valid.shape != predicted.shape[:-1]:
        raise ValueError(
            f"target joints/mask shapes must be {tuple(predicted.shape)}/"
            f"{tuple(predicted.shape[:-1])}, got {tuple(target_joints.shape)}/"
            f"{tuple(valid.shape)}"
        )
    valid = valid.to(device=predicted.device, dtype=torch.bool)
    target_joints = target_joints.to(device=predicted.device, dtype=predicted.dtype)
    valid = valid & torch.isfinite(target_joints).all(dim=-1)
    if not valid.any():
        raise ValueError("external 3D joint target has no valid points")
    error = (predicted - target_joints).abs()
    return error.masked_select(valid.unsqueeze(-1)).mean()


def acceleration_loss(
    prediction: ManoFlowState, target: ManoFlowState, mano: ManoWrapper
) -> Tensor:
    """Mean L1 error between unit-time second differences of joints."""
    prediction_joints = mano(prediction)
    target_joints = mano(target)
    prediction_acceleration = (
        prediction_joints[:, :, 2:] - 2 * prediction_joints[:, :, 1:-1] + prediction_joints[:, :, :-2]
    )
    target_acceleration = (
        target_joints[:, :, 2:] - 2 * target_joints[:, :, 1:-1] + target_joints[:, :, :-2]
    )
    return (prediction_acceleration - target_acceleration).abs().mean()


def camera_pose_loss(prediction: CameraFlowState, target: CameraFlowState) -> Tensor:
    """Mean absolute camera pose error (geodesic rotation plus L1 translation)."""
    prediction_rotation = rotation_6d_to_matrix(prediction.rotation)
    target_rotation = rotation_6d_to_matrix(target.rotation)
    rotation_error = geodesic_distance(prediction_rotation, target_rotation).mean()
    translation_error = (prediction.translation - target.translation).abs().mean()
    return rotation_error + translation_error


def camera_trajectory_loss(prediction: CameraFlowState, target: CameraFlowState) -> Tensor:
    """Mean consecutive-camera error (geodesic rotation plus L1 displacement)."""
    prediction_rotation = rotation_6d_to_matrix(prediction.rotation)
    target_rotation = rotation_6d_to_matrix(target.rotation)
    prediction_delta = prediction_rotation[:, :-1].transpose(-2, -1) @ prediction_rotation[:, 1:]
    target_delta = target_rotation[:, :-1].transpose(-2, -1) @ target_rotation[:, 1:]
    rotation_error = geodesic_distance(prediction_delta, target_delta).mean()
    prediction_displacement = prediction.translation[:, 1:] - prediction.translation[:, :-1]
    target_displacement = target.translation[:, 1:] - target.translation[:, :-1]
    translation_error = (prediction_displacement - target_displacement).abs().mean()
    return rotation_error + translation_error
