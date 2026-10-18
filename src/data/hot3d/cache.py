"""HaMeR per-frame and evaluation-only VGGT fixed-window token caches."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch
from torch import Tensor

_HANDS = ("left", "right")


def _make_hamer_crop(cfg, image_bgr: np.ndarray, box: np.ndarray, side: int) -> Tensor:
    import cv2
    from scipy.ndimage import gaussian_filter
    x1, y1, x2, y2 = box.astype(np.float32)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = 2.0 * (x2 - x1), 2.0 * (y2 - y1)
    bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)
    if bbox_shape is not None:
        target_w, target_h = bbox_shape
        if height / width < target_h / target_w:
            height = max(width * target_h / target_w, height)
        else:
            width = max(height * target_w / target_h, width)
    bbox_size = max(width, height)
    output_size = int(cfg.MODEL.IMAGE_SIZE)
    downsampling_factor = bbox_size / output_size / 2.0
    if downsampling_factor > 1.1:
        image_bgr = gaussian_filter(
            image_bgr.astype(np.float64),
            sigma=((downsampling_factor - 1) / 2,) * 2 + (0,), mode="nearest",
        )
    if side == 0:
        image_bgr = image_bgr[:, ::-1]
        center_x = image_bgr.shape[1] - center_x - 1
    source = np.float32([[center_x, center_y], [center_x, center_y + bbox_size / 2], [center_x + bbox_size / 2, center_y]])
    destination = np.float32([[output_size / 2, output_size / 2], [output_size / 2, output_size], [output_size, output_size / 2]])
    crop = cv2.warpAffine(image_bgr, cv2.getAffineTransform(source, destination), (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    crop = crop[:, :, ::-1].copy().astype(np.float32) / 255.0
    crop = (crop - np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)) / np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)
    return torch.from_numpy(crop.transpose(2, 0, 1)).float()


class HaMeRTokenCache:
    def __init__(self, *, cache_root: str | Path, hamer, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.cache_root = Path(cache_root)
        self.hamer = hamer
        self.cfg = hamer.model_cfg
        self.batch_size = batch_size

    def _path(self, clip_id: str, frame: int, hand: str) -> Path:
        return self.cache_root / "hamer" / clip_id / f"{frame:06d}_{hand}.pt"

    def _load(self, path: Path) -> Tensor | None:
        if not path.is_file():
            return None
        try:
            token = torch.load(path, map_location="cpu", weights_only=True)["token"]
            token = torch.as_tensor(token, dtype=torch.float32)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None
        if token.shape != (1024,) or not torch.isfinite(token).all():
            return None
        return token

    def _save(self, path: Path, token: Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save({"token": token.detach().cpu().clone()}, temporary)
        temporary.replace(path)

    def get_tokens(self, dataset, boxes: Tensor, valid_boxes: Tensor) -> tuple[Tensor, Tensor]:
        """Return tokens using externally predicted hand boxes only."""
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        valid_boxes = torch.as_tensor(valid_boxes, dtype=torch.bool)
        expected = (2, len(dataset.frames), 4)
        if boxes.shape != expected or valid_boxes.shape != expected[:2]:
            raise ValueError(f"handbox tensors must have shapes {expected} and {expected[:2]}")
        tokens = torch.zeros(2, len(dataset.frames), 1024)
        valid = torch.zeros(2, len(dataset.frames), dtype=torch.bool)
        missing = [
            (hand, frame)
            for hi, hand in enumerate(_HANDS)
            for ti, frame in enumerate(dataset.frames)
            if bool(valid_boxes[hi, ti]) and not self._path(dataset.clip_id, frame, hand).is_file()
        ]
        if missing and not getattr(self.hamer, "offline", False):
            crops, indices = [], []
            for hand, frame in missing:
                hi = _HANDS.index(hand)
                ti = dataset.frames.index(frame)
                image_bgr = np.ascontiguousarray(dataset.image(frame)[:, :, ::-1])
                crops.append(_make_hamer_crop(self.cfg, image_bgr, boxes[hi, ti].numpy(), hi))
                indices.append((hand, frame))
            device = next(self.hamer.parameters()).device
            batch = torch.stack(crops)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                generated = self.hamer(batch[start:end].to(device))
                for (hand, frame), token in zip(indices[start:end], generated):
                    hi = _HANDS.index(hand)
                    ti = dataset.frames.index(frame)
                    token = token.detach().cpu()
                    tokens[hi, ti] = token
                    valid[hi, ti] = True
                    self._save(self._path(dataset.clip_id, frame, hand), token)
        for hi, hand in enumerate(_HANDS):
            for ti, frame in enumerate(dataset.frames):
                if not valid_boxes[hi, ti] or valid[hi, ti]:
                    continue
                token = self._load(self._path(dataset.clip_id, frame, hand))
                if token is not None:
                    tokens[hi, ti] = token
                    valid[hi, ti] = True
        return tokens, valid

    def get_tokens_batch(self, datasets, boxes_list, valid_list):
        """Extract all missing hand crops from several windows in shared batches."""
        states = []
        missing = []
        for dataset_index, (dataset, boxes, valid_boxes) in enumerate(
            zip(datasets, boxes_list, valid_list)
        ):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            valid_boxes = torch.as_tensor(valid_boxes, dtype=torch.bool)
            expected = (2, len(dataset.frames), 4)
            if boxes.shape != expected or valid_boxes.shape != expected[:2]:
                raise ValueError(f"handbox tensors must have shapes {expected} and {expected[:2]}")
            tokens = torch.zeros(2, len(dataset.frames), 1024)
            valid = torch.zeros(2, len(dataset.frames), dtype=torch.bool)
            states.append((tokens, valid, boxes, valid_boxes))
            missing.extend(
                (dataset_index, hand, frame)
                for hi, hand in enumerate(_HANDS)
                for ti, frame in enumerate(dataset.frames)
                if bool(valid_boxes[hi, ti])
                and not self._path(dataset.clip_id, frame, hand).is_file()
            )
        if missing and not getattr(self.hamer, "offline", False):
            crops, indices = [], []
            for dataset_index, hand, frame in missing:
                dataset = datasets[dataset_index]
                boxes = states[dataset_index][2]
                hi = _HANDS.index(hand)
                ti = dataset.frames.index(frame)
                image_bgr = np.ascontiguousarray(dataset.image(frame)[:, :, ::-1])
                crops.append(_make_hamer_crop(self.cfg, image_bgr, boxes[hi, ti].numpy(), hi))
                indices.append((dataset_index, hand, frame))
            device = next(self.hamer.parameters()).device
            batch = torch.stack(crops)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                generated = self.hamer(batch[start:end].to(device))
                for (dataset_index, hand, frame), token in zip(indices[start:end], generated):
                    dataset = datasets[dataset_index]
                    hi = _HANDS.index(hand)
                    ti = dataset.frames.index(frame)
                    token = token.detach().cpu()
                    states[dataset_index][0][hi, ti] = token
                    states[dataset_index][1][hi, ti] = True
                    self._save(self._path(dataset.clip_id, frame, hand), token)
        outputs = []
        for dataset_index, dataset in enumerate(datasets):
            tokens, valid, _boxes, valid_boxes = states[dataset_index]
            for hi, hand in enumerate(_HANDS):
                for ti, frame in enumerate(dataset.frames):
                    if not valid_boxes[hi, ti] or valid[hi, ti]:
                        continue
                    token = self._load(self._path(dataset.clip_id, frame, hand))
                    if token is not None:
                        tokens[hi, ti] = token
                        valid[hi, ti] = True
            outputs.append((tokens, valid))
        return outputs


class VGGTOmegaTokenCache:
    """Evaluation-only cache for one token artifact per fixed window."""

    def __init__(self, *, cache_root: str | Path, vggt) -> None:
        self.cache_root = Path(cache_root)
        self.vggt = vggt

    def _path(self, clip_id: str, frames: list[int]) -> Path:
        if frames != list(range(64)) and frames != list(range(64, 128)):
            raise ValueError("VGGT eval cache only supports frames 0..63 and 64..127")
        return self.cache_root / "vggt" / clip_id / f"{frames[0]:06d}-{frames[-1]:06d}.pt"

    @torch.no_grad()
    def get_tokens(self, dataset) -> Tensor:
        path = self._path(dataset.clip_id, dataset.frames)
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)["tokens"]
        if self.vggt is None:
            raise RuntimeError(f"missing VGGT cache for {dataset.clip_id} frames {dataset.frames[0]}..{dataset.frames[-1]}")
        images = torch.stack([torch.from_numpy(dataset.image(frame)).permute(2, 0, 1).float() / 255.0 for frame in dataset.frames]).unsqueeze(0)
        device = next(self.vggt.parameters()).device
        tokens = self.vggt(images.to(device)).detach().cpu().float().clone()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"tokens": tokens}, temporary)
        temporary.replace(path)
        return tokens
