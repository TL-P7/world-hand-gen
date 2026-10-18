"""Frozen MANO wrapper mapping flow state to world-space joints."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config import MANO_ROOT
from src.geometry.rotation import rotation_6d_to_matrix
from src.model.flow.state import ManoFlowState


class ManoWrapper(nn.Module):
    """Map ``ManoFlowState`` (6D rotations) to ``[B,2,T,16,3]`` joints.

    MANO layers are loaded lazily on the input device/dtype. Their parameters
    are frozen, but decoding remains differentiable with respect to the input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._layers = nn.ModuleDict()

    def _ensure_layers(self, reference: Tensor) -> None:
        if self._layers:
            return
        import smplx

        for hand in ("left", "right"):
            self._layers[hand] = smplx.MANOLayer(
                str(MANO_ROOT / "mano"), is_rhand=hand == "right", use_pca=False,
                flat_hand_mean=True,
            ).to(device=reference.device, dtype=reference.dtype).requires_grad_(False).eval()

    def _decode(self, mano: ManoFlowState, *, include_vertices: bool):
        self._ensure_layers(mano.translation)
        batch, _, time = mano.translation.shape[:3]

        root_matrix = rotation_6d_to_matrix(mano.root_rotation)
        local_matrix = rotation_6d_to_matrix(mano.local_rotation)

        joints = []
        vertices = []
        faces = []
        for hand_index, hand in enumerate(("left", "right")):
            layer = self._layers[hand]
            root = root_matrix[:, hand_index].reshape(-1, 1, 3, 3)
            local = local_matrix[:, hand_index].reshape(-1, 15, 3, 3)
            transl = mano.translation[:, hand_index].reshape(-1, 3)
            output = layer(
                global_orient=root,
                hand_pose=local,
                betas=root.new_zeros(root.shape[0], 10),
                transl=transl,
            )
            joints.append(output.joints.reshape(batch, time, 16, 3))
            if include_vertices:
                if not hasattr(output, "vertices"):
                    raise RuntimeError("MANO layer output does not expose vertices")
                vertices.append(output.vertices.reshape(batch, time, -1, 3))
                layer_faces = getattr(layer, "faces_tensor", None)
                if layer_faces is None:
                    layer_faces = getattr(layer, "faces", None)
                if layer_faces is None:
                    raise RuntimeError("MANO layer does not expose mesh faces")
                faces.append(torch.as_tensor(layer_faces, device=output.vertices.device, dtype=torch.long))
        joint_tensor = torch.stack(joints, dim=1)
        if not include_vertices:
            return joint_tensor, None, None
        return joint_tensor, torch.stack(vertices, dim=1), tuple(faces)

    def forward(self, mano: ManoFlowState) -> Tensor:
        """Decode MANO joints without allocating mesh vertices."""
        return self._decode(mano, include_vertices=False)[0]

    def forward_mesh(self, mano: ManoFlowState):
        """Decode MANO vertices, joints, and per-hand triangle faces.

        Returns ``(vertices, joints, faces)`` where vertices has shape
        ``[B,2,T,V,3]``, joints has shape ``[B,2,T,16,3]``, and faces is a
        two-element tuple of integer tensors. MANO layers remain lazily loaded
        here, just as they are for :meth:`forward`.
        """
        joints, vertices, faces = self._decode(mano, include_vertices=True)
        return vertices, joints, faces
