"""Condition token projection for HaMeR, VGGT, bbox, and ViTPose keypoints."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .state import ConditionTokens


class ConditionEncoder(nn.Module):
    """Project raw detector/model conditions into model-dimension tokens."""

    def __init__(
        self,
        *,
        hand_dim: int = 1024,
        camera_dim: int = 2048,
        model_dim: int = 512,
        bbox_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.hamer = nn.Sequential(nn.LayerNorm(hand_dim), nn.Linear(hand_dim, model_dim))
        self.vggt = nn.Sequential(nn.LayerNorm(camera_dim), nn.Linear(camera_dim, model_dim))
        self.bbox = nn.Sequential(
            nn.Linear(4, bbox_hidden),
            nn.SiLU(),
            nn.Linear(bbox_hidden, model_dim),
        )
        self.keypoints = nn.Sequential(nn.LayerNorm(63), nn.Linear(63, model_dim))
        self.hamer_missing = nn.Parameter(torch.zeros(model_dim))
        self.vggt_missing = nn.Parameter(torch.zeros(model_dim))
        self.bbox_missing = nn.Parameter(torch.zeros(model_dim))
        self.keypoint_missing = nn.Parameter(torch.zeros(model_dim))

    def forward(
        self,
        hamer: Tensor,  # [B,2,T,1024]
        vggt: Tensor,  # [B,T,17,2048]
        bbox: Tensor,  # [B,2,T,4]
        hamer_mask: Tensor,  # bool [B,2,T]
        vggt_mask: Tensor,  # bool [B,T]
        bbox_mask: Tensor,  # bool [B,2,T]
        keypoints: Tensor | None = None,  # [B,2,T,21,3]
    ) -> ConditionTokens:
        batch = hamer.shape[0]
        time = hamer.shape[2]
        if keypoints is None:
            keypoints = torch.zeros(
                batch, 2, time, 21, 3, device=hamer.device, dtype=hamer.dtype
            )
        if keypoints.shape != (batch, 2, time, 21, 3):
            raise ValueError("keypoints must have shape [B,2,T,21,3]")

        hamer_tokens = self.hamer(hamer).reshape(batch, 2 * time, -1)
        hamer_flat_mask = hamer_mask.reshape(batch, 2 * time)
        hamer_tokens = torch.where(
            hamer_flat_mask.unsqueeze(-1), hamer_tokens, self.hamer_missing
        )

        vggt_tokens = self.vggt(vggt).reshape(batch, time * 17, -1)
        vggt_flat_mask = (
            vggt_mask.unsqueeze(-1).expand(batch, time, 17).reshape(batch, time * 17)
        )
        vggt_tokens = torch.where(
            vggt_flat_mask.unsqueeze(-1), vggt_tokens, self.vggt_missing
        )

        bbox_tokens = self.bbox(bbox).reshape(batch, 2 * time, -1)
        bbox_flat_mask = bbox_mask.reshape(batch, 2 * time)
        bbox_tokens = torch.where(
            bbox_flat_mask.unsqueeze(-1), bbox_tokens, self.bbox_missing
        )

        keypoint_values = keypoints.reshape(batch, 2 * time, 63)
        keypoint_tokens = self.keypoints(keypoint_values)
        # A missing hand/frame is represented by zero confidence for all points.
        keypoint_missing = keypoints[..., 2].reshape(batch, 2 * time, 21).max(dim=-1).values <= 0
        keypoint_tokens = torch.where(
            keypoint_missing.unsqueeze(-1), self.keypoint_missing, keypoint_tokens
        )

        return ConditionTokens(
            hamer=hamer_tokens,
            vggt=vggt_tokens,
            bbox=bbox_tokens,
            keypoints=keypoint_tokens,
        )
