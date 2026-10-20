from __future__ import annotations

import pytest
import torch
from torch import nn

from src.geometry.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
from src.model.flow.state import ManoFlowState
from src.model.wrappers.mano_fit import fit_joints_to_mano


_PUBLIC_FROM_MANO = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)


class _FakeMano(nn.Module):
    """Differentiable decoder with native joints carrying visible indices."""

    def forward(self, state: ManoFlowState) -> torch.Tensor:
        batch, hands, time = state.translation.shape[:3]
        root = rotation_6d_to_matrix(state.root_rotation)[..., :, 0]
        local = rotation_6d_to_matrix(state.local_rotation)[..., :, 0]
        native = torch.zeros(batch, hands, time, 16, 3, device=state.translation.device)
        native[..., 0, :] = root
        native[..., 1:16, :] = local
        return native + state.translation.unsqueeze(-2)


class _DirectFakeMano(nn.Module):
    """fit_egodex-style decoder returning OpenPose-21 joints."""

    def forward(self, global_orient: torch.Tensor, hand_pose: torch.Tensor):
        batch = global_orient.shape[0]
        joints = torch.zeros(batch, 21, 3, device=global_orient.device)
        joints[:, 0] = global_orient[:, 0, :, 0]
        joints[:, 1:16] = hand_pose[:, :, 0]
        joints[:, 16:20] = hand_pose[:, 11:15, 1]
        return type("Output", (), {"joints": joints})()


def _identity_state(batch: int = 1, hands: int = 2, time: int = 1) -> ManoFlowState:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return ManoFlowState(
        translation=torch.zeros(batch, hands, time, 3),
        root_rotation=identity.expand(batch, hands, time, 6).clone(),
        local_rotation=identity.expand(batch, hands, time, 15, 6).clone(),
    )


def test_fit_accepts_fit_egodex_style_decoder() -> None:
    model = _DirectFakeMano()
    target = torch.zeros(4, 16, 3)
    fitted = fit_joints_to_mano(model, target, steps=1)
    assert fitted.translation.shape == (4, 1, 1, 3)
    assert torch.isfinite(fitted.local_rotation).all()


def test_fit_accepts_single_hand_batch() -> None:
    model = _FakeMano()
    initial = _identity_state(batch=4, hands=1)
    target = model(initial)[..., list(_PUBLIC_FROM_MANO), :].reshape(4, 16, 3)
    fitted = fit_joints_to_mano(model, target, steps=0)

    assert fitted.translation.shape == (4, 1, 1, 3)
    assert fitted.root_rotation.shape == (4, 1, 1, 6)
    assert fitted.local_rotation.shape == (4, 1, 1, 15, 6)


def test_fit_uses_public_joint_order_and_returns_flow_state() -> None:
    model = _FakeMano()
    initial = _identity_state()
    native = model(initial)
    public = native[..., list(_PUBLIC_FROM_MANO), :]
    fitted = fit_joints_to_mano(model, public, initial_state=initial, steps=0)

    assert isinstance(fitted, ManoFlowState)
    assert fitted.translation.shape == (1, 2, 1, 3)
    assert fitted.root_rotation.shape == (1, 2, 1, 6)
    assert fitted.local_rotation.shape == (1, 2, 1, 15, 6)
    assert torch.isfinite(fitted.translation).all()
    assert torch.isfinite(fitted.root_rotation).all()
    assert torch.isfinite(fitted.local_rotation).all()
    assert torch.allclose(model(fitted)[..., list(_PUBLIC_FROM_MANO), :], public)


def test_fit_reduces_masked_joint_error() -> None:
    model = _FakeMano()
    initial = _identity_state()
    target_state = _identity_state()
    target_state.translation[..., 0] = 0.4
    target_native = model(target_state)
    target = target_native[..., list(_PUBLIC_FROM_MANO), :]

    before = (model(initial)[..., list(_PUBLIC_FROM_MANO), :] - target).norm(dim=-1).mean()
    fitted = fit_joints_to_mano(model, target, initial_state=initial, steps=8, lr=0.05)
    after = (model(fitted)[..., list(_PUBLIC_FROM_MANO), :] - target).norm(dim=-1).mean()

    assert after < before


def test_fit_masks_nan_joints_and_rejects_empty_mask() -> None:
    model = _FakeMano()
    initial = _identity_state()
    target = model(initial)[..., list(_PUBLIC_FROM_MANO), :]
    target[..., 0, 0] = float("nan")
    valid = torch.ones(1, 2, 1, 16, dtype=torch.bool)
    valid[..., 0] = False
    fitted = fit_joints_to_mano(model, target, initial_state=initial, joint_valid=valid, steps=1)
    assert torch.isfinite(fitted.translation).all()

    with pytest.raises(ValueError, match="no valid points"):
        fit_joints_to_mano(model, target, initial_state=initial, joint_valid=torch.zeros_like(valid))
