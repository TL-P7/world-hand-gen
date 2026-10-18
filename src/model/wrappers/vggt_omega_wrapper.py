"""Frozen VGGT-Omega wrapper exposing camera/register tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VGGTOmegaCameraTokenWrapper(nn.Module):
    """Load VGGT-Omega and return one camera plus 16 register tokens per frame."""

    def __init__(
        self,
        *,
        repository: str | Path,
        checkpoint: str | Path,
        image_resolution: int = 512,
    ) -> None:
        super().__init__()
        repository = Path(repository).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(f"VGGT-Omega repository does not exist: {repository}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint}")
        if not isinstance(image_resolution, int) or image_resolution < 1:
            raise ValueError("image_resolution must be a positive integer")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))

        from vggt_omega.models.vggt_omega import VGGTOmega

        self.checkpoint = str(checkpoint)
        self.image_resolution = image_resolution
        self.model = VGGTOmega(enable_depth=False, enable_alignment=False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = self._state_dict(payload)
        state_dict = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("dense_head.")
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.train(False)

    @staticmethod
    def _state_dict(payload: object) -> dict[str, Tensor]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model", "module"):
                value = payload.get(key)
                if isinstance(value, dict) and value and all(
                    isinstance(item, Tensor) for item in value.values()
                ):
                    return value
            if payload and all(isinstance(item, Tensor) for item in payload.values()):
                return payload  # type: ignore[return-value]
        raise ValueError("VGGT checkpoint must contain a tensor state_dict")

    def train(self, mode: bool = True) -> VGGTOmegaCameraTokenWrapper:
        """Keep this inference-only wrapper in evaluation mode."""
        super().train(False)
        return self

    @torch.inference_mode()
    def _predict(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("VGGT images must have shape [B,T,3,H,W]")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("VGGT images must be finite floating point")
        if (images < 0).any() or (images > 1).any():
            raise ValueError("VGGT images must lie in [0,1]")
        batch, time = images.shape[:2]
        if tuple(images.shape[-2:]) != (self.image_resolution, self.image_resolution):
            images = F.interpolate(
                images.flatten(0, 1),
                size=(self.image_resolution, self.image_resolution),
                mode="bicubic",
                align_corners=False,
            ).reshape(batch, time, 3, self.image_resolution, self.image_resolution)
        return self.model(images)

    def forward(self, images: Tensor) -> Tensor:
        predictions = self._predict(images)
        tokens = predictions.get("camera_and_register_tokens")
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 4
            or tokens.shape[2] != 17
            or tokens.shape[3] != 2048
        ):
            raise ValueError(
                "VGGT output camera_and_register_tokens must be [B,T,17,2048]"
            )
        if not torch.isfinite(tokens).all():
            raise ValueError("VGGT camera/register tokens must be finite")
        return tokens.contiguous()
