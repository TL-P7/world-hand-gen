from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.model.flow.state import ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("faces_tensor", torch.tensor([[0, 1, 2], [0, 2, 3]]))

    def forward(self, *, global_orient, hand_pose, betas, transl):
        batch = transl.shape[0]
        vertices = torch.zeros(batch, 4, 3, device=transl.device, dtype=transl.dtype)
        vertices[..., 0, :] = transl
        joints = torch.zeros(batch, 16, 3, device=transl.device, dtype=transl.dtype)
        joints[..., 0, :] = transl
        return SimpleNamespace(vertices=vertices, joints=joints)


def _state():
    return ManoFlowState(
        translation=torch.randn(1, 2, 3, 3),
        root_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(1, 2, 3, 6).clone(),
        local_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(1, 2, 3, 15, 6).clone(),
    )


def test_forward_mesh_returns_vertices_joints_and_faces_without_model_loading(monkeypatch):
    wrapper = ManoWrapper()
    wrapper._layers = nn.ModuleDict({"left": _FakeLayer(), "right": _FakeLayer()})
    monkeypatch.setattr(wrapper, "_ensure_layers", lambda reference: None)

    vertices, joints, faces = wrapper.forward_mesh(_state())

    assert vertices.shape == (1, 2, 3, 4, 3)
    assert joints.shape == (1, 2, 3, 16, 3)
    assert len(faces) == 2
    assert all(face.shape == (2, 3) and face.dtype == torch.long for face in faces)
