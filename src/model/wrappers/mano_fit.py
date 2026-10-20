"""Batch fitting of MANO pose and translation from 3D joints."""

from __future__ import annotations

import inspect

import torch
from torch import Tensor

from src.geometry.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from src.model.flow.state import ManoFlowState


# Native MANO-16 order is wrist, index, middle, little, ring, thumb.
# Public order is wrist, thumb, index, middle, ring, little.
_PUBLIC_FROM_MANO = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)
# fit_egodex's decoder exposes OpenPose-21 joints; its 16-point target
# order is wrist, thumb (3), index (3), middle (3), ring (3), little (3).
_OP_FROM_MANO_21 = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)


def _public_joints(native_joints: Tensor) -> Tensor:
    return native_joints[..., list(_PUBLIC_FROM_MANO), :]


def _validate_state(state: ManoFlowState, shape: tuple[int, int, int]) -> None:
    batch, hands, time = shape
    expected = {
        "translation": (batch, hands, time, 3),
        "root_rotation": (batch, hands, time, 6),
        "local_rotation": (batch, hands, time, 15, 6),
    }
    for name, value in (
        ("translation", state.translation),
        ("root_rotation", state.root_rotation),
        ("local_rotation", state.local_rotation),
    ):
        if tuple(value.shape) != expected[name]:
            raise ValueError(
                f"initial_state.{name} must have shape {expected[name]}, got {tuple(value.shape)}"
            )


def _model_reference(mano_model, target: Tensor) -> tuple[torch.device, torch.dtype]:
    parameters = getattr(mano_model, "parameters", None)
    if parameters is not None:
        parameter = next(iter(parameters()), None)
        if parameter is not None:
            return parameter.device, parameter.dtype
    return target.device, target.dtype


def _uses_direct_decoder(mano_model) -> bool:
    """Return whether the model follows the fit_egodex keyword API."""
    try:
        parameters = inspect.signature(mano_model.forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "global_orient" in parameters and "hand_pose" in parameters


def _decode(
    mano_model,
    state: ManoFlowState,
    root_axis: Tensor,
    local_axis: Tensor,
    translation: Tensor,
    *,
    direct_decoder: bool,
) -> Tensor:
    """Decode public-order joints as ``[B,2,T,16,3]``."""
    if not direct_decoder:
        return _public_joints(mano_model(state))

    batch, hands, time = translation.shape[:3]
    root = axis_angle_to_matrix(root_axis).reshape(-1, 1, 3, 3)
    local = axis_angle_to_matrix(local_axis).reshape(-1, 15, 3, 3)
    output = mano_model(global_orient=root, hand_pose=local)
    native = output.joints if hasattr(output, "joints") else output
    if native.ndim != 3 or native.shape[0] != batch * hands * time:
        raise ValueError(
            "direct MANO decoder must return joints with shape [B,16|21,3] "
            f"for a flattened batch, got {tuple(native.shape)}"
        )
    if native.shape[-2] == 16:
        public = _public_joints(native)
    elif native.shape[-2] >= 20:
        public = native[:, list(_OP_FROM_MANO_21), :]
    else:
        raise ValueError(f"direct MANO decoder returned {native.shape[-2]} joints; expected 16 or 21")
    return (public + translation.reshape(-1, 1, 3)).reshape(batch, hands, time, 16, 3)


def fit_joints_to_mano(
    mano_model,
    joints3d: Tensor,
    *,
    initial_state: ManoFlowState | None = None,
    joint_valid: Tensor | None = None,
    steps: int = 20,
    lr: float = 0.05,
    pose_regularization: float = 1e-2,
) -> ManoFlowState:
    """Fit MANO pose and translation to a batch of public-order 3D joints.

    ``joints3d`` accepts either ``[B,16,3]`` for a single-hand batch (the
    fit_egodex form) or ``[B,2,T,16,3]`` for the project's two-hand windows.
    A single-hand input returns a ``ManoFlowState`` with shape ``[B,1,1,...]``.
    The decoder may be a project ``ManoWrapper`` or a fit_egodex-style module
    accepting ``global_orient`` and ``hand_pose`` rotation matrices. MANO
    shape/betas remain fixed by the supplied model.
    """
    if joints3d.ndim == 3 and joints3d.shape[-2:] == (16, 3):
        target = joints3d.unsqueeze(1).unsqueeze(1)
    elif joints3d.ndim == 5 and joints3d.shape[1] == 2 and joints3d.shape[-2:] == (16, 3):
        target = joints3d
    else:
        raise ValueError("joints3d must have shape [B,16,3] or [B,2,T,16,3]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if pose_regularization < 0:
        raise ValueError("pose_regularization must be nonnegative")

    batch, hands, time = target.shape[:3]
    if joint_valid is None:
        valid = torch.isfinite(target).all(dim=-1)
    else:
        expected_valid_shape = (batch, 16) if joints3d.ndim == 3 else (batch, hands, time, 16)
        if tuple(joint_valid.shape) != expected_valid_shape:
            raise ValueError(
                f"joint_valid must have shape {expected_valid_shape}, got {tuple(joint_valid.shape)}"
            )
        valid = joint_valid.unsqueeze(1).unsqueeze(1) if joints3d.ndim == 3 else joint_valid
        valid = valid.to(dtype=torch.bool) & torch.isfinite(target).all(dim=-1)
    if not valid.any():
        raise ValueError("joints3d has no valid points")

    device, dtype = _model_reference(mano_model, target)
    target = torch.nan_to_num(target.detach().to(device=device, dtype=dtype))
    valid = valid.to(device=device, dtype=torch.bool)
    direct_decoder = _uses_direct_decoder(mano_model)

    if initial_state is None:
        root_axis = torch.zeros(batch, hands, time, 3, device=device, dtype=dtype)
        local_axis = torch.zeros(batch, hands, time, 15, 3, device=device, dtype=dtype)
        translation = torch.zeros(batch, hands, time, 3, device=device, dtype=dtype)
    else:
        _validate_state(initial_state, (batch, hands, time))
        initial = initial_state.map(lambda value: value.detach().to(device=device, dtype=dtype))
        root_axis = matrix_to_axis_angle(rotation_6d_to_matrix(initial.root_rotation))
        local_axis = matrix_to_axis_angle(rotation_6d_to_matrix(initial.local_rotation))
        translation = initial.translation.clone()

    if initial_state is None:
        with torch.no_grad():
            identity_state = ManoFlowState(
                translation=translation,
                root_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(root_axis)),
                local_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(local_axis)),
            )
            zero_public = _decode(
                mano_model, identity_state, root_axis, local_axis, translation,
                direct_decoder=direct_decoder,
            )
            translation.copy_(target[..., 0, :] - zero_public[..., 0, :])

    root_axis = root_axis.clone().requires_grad_(True)
    local_axis = local_axis.clone().requires_grad_(True)
    translation = translation.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([root_axis, local_axis, translation], lr=lr)

    was_training = getattr(mano_model, "training", None)
    if hasattr(mano_model, "eval"):
        mano_model.eval()
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            state = ManoFlowState(
                translation=translation,
                root_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(root_axis)),
                local_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(local_axis)),
            )
            predicted = _decode(
                mano_model, state, root_axis, local_axis, translation,
                direct_decoder=direct_decoder,
            )
            if tuple(predicted.shape) != tuple(target.shape):
                raise ValueError(
                    f"MANO decoder must produce public joints with shape {tuple(target.shape)}, "
                    f"got {tuple(predicted.shape)}"
                )
            loss_j3d = (predicted - target).norm(dim=-1).masked_select(valid).mean()
            loss = loss_j3d + pose_regularization * local_axis.square().mean()
            loss.backward()
            optimizer.step()
    finally:
        if was_training is True and hasattr(mano_model, "train"):
            mano_model.train()

    with torch.no_grad():
        return ManoFlowState(
            translation=translation.detach(),
            root_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(root_axis.detach())),
            local_rotation=matrix_to_rotation_6d(axis_angle_to_matrix(local_axis.detach())),
        )
