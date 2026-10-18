"""Dataset-neutral flow state/model for direct 3D hand-joint supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
from torch import Tensor, nn

from .time import ScalarTimeEncoder, sinusoidal_embedding

T = TypeVar("T", bound="JointFlowState")


@dataclass
class JointFlowState:
    """Public-order 3D joints with shape ``[B,2,T,J,3]``."""

    joints: Tensor

    def map(self: T, function: Callable[[Tensor], Tensor]) -> T:
        return type(self)(function(self.joints))


class JointFlowModel(nn.Module):
    """Small conditional-free flow model for direct joint convergence checks."""

    def __init__(self, joints: int = 16, model_dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        self.joints = joints
        self.model_dim = model_dim
        self.input = nn.Linear(3, model_dim)
        self.time = ScalarTimeEncoder(model_dim)
        self.hand = nn.Parameter(torch.zeros(1, 2, 1, 1, model_dim))
        self.joint = nn.Parameter(torch.zeros(1, 1, 1, joints, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=4 * model_dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: JointFlowState, flow_time: Tensor) -> JointFlowState:
        joints = state.joints
        if joints.ndim != 5 or joints.shape[1] != 2 or joints.shape[-1] != 3:
            raise ValueError("joint state must have shape [B,2,T,J,3]")
        batch, hands, time, joints_count, _ = joints.shape
        if joints_count != self.joints or flow_time.shape != (batch,):
            raise ValueError("joint state/time shape mismatch")
        tokens = self.input(joints) + self.hand + self.joint
        tokens = tokens + self.time(flow_time).reshape(batch, 1, 1, 1, self.model_dim)
        tokens = tokens.reshape(batch, hands * time * joints_count, self.model_dim)
        tokens = self.encoder(tokens)
        output = self.output(self.norm(tokens)).reshape(batch, hands, time, joints_count, 3)
        return JointFlowState(output)
