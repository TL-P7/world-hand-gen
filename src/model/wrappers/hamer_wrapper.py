"""Frozen HaMeR wrapper exposing decoder query tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from src.config import HAMER_CHECKPOINT, HAMER_REPOSITORY


class HaMeRDecoderTokenWrapper(nn.Module):
    """Load HaMeR and expose its final 1024-dimensional decoder token.

    Args:
        repository: Local HaMeR source checkout.
        checkpoint: HaMeR checkpoint file.

    The wrapper accepts HaMeR-normalized crops with shape ``[N, 3, 256, 256]``.
    Crop construction and bounding-box bookkeeping remain outside this class.
    """

    def __init__(
        self, *, repository: str | Path = HAMER_REPOSITORY,
        checkpoint: str | Path = HAMER_CHECKPOINT,
    ) -> None:
        super().__init__()
        repository = Path(repository).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(f"HaMeR repository does not exist: {repository}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"HaMeR checkpoint does not exist: {checkpoint}")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))

        from hamer.configs import get_config
        from hamer.models import HAMER

        config_path = checkpoint.parent.parent / "model_config.yaml"
        model_cfg = get_config(str(config_path), update_cachedir=True)
        model_cfg.defrost()
        data_root = repository / "_DATA" / "data"
        model_cfg.MANO.DATA_DIR = str(data_root)
        model_cfg.MANO.MODEL_PATH = str(data_root / "mano")
        model_cfg.MANO.MEAN_PARAMS = str(data_root / "mano_mean_params.npz")
        if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

        self.checkpoint = str(checkpoint)
        self.model_cfg = model_cfg
        self.model = HAMER.load_from_checkpoint(
            str(checkpoint), strict=False, cfg=model_cfg
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.train(False)

    def train(self, mode: bool = True) -> HaMeRDecoderTokenWrapper:
        """Keep this inference-only wrapper in evaluation mode."""
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, crops: Tensor) -> Tensor:
        if crops.ndim != 4 or tuple(crops.shape[1:]) != (3, 256, 256):
            raise ValueError("HaMeR crops must have shape [N,3,256,256]")
        features = self.model.backbone(crops[:, :, :, 32:-32])
        context = features.flatten(2).transpose(1, 2)
        query = torch.zeros(
            crops.shape[0], 1, 1, device=crops.device, dtype=features.dtype
        )
        decoder = self.model.mano_head.transformer(query, context=context)
        token = decoder[:, 0]
        if token.shape != (crops.shape[0], 1024) or not torch.isfinite(token).all():
            raise ValueError("HaMeR decoder query must be finite [N,1024]")
        return token
