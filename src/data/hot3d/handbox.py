"""Per-frame ViTDet/ViTPose detection cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class HandBoxCache:
    """Cache ViTPose hand keypoints and boxes independently of HaMeR tokens."""

    def __init__(self, *, cache_root: str | Path, detector, batch_size: int = 4) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.cache_root = Path(cache_root)
        self.detector = detector
        self.batch_size = batch_size

    def _path(self, clip_id: str, frame: int) -> Path:
        return self.cache_root / "handbox-v2" / clip_id / f"{frame:06d}.pt"

    @staticmethod
    def _validate_boxes(record: dict, path: Path) -> tuple[Tensor, Tensor]:
        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32)
        valid = torch.as_tensor(record.get("box_valid", record.get("valid")), dtype=torch.bool)
        if boxes.shape != (2, 4) or valid.shape != (2,):
            raise ValueError(f"invalid handbox cache: {path}")
        if valid.any() and not torch.isfinite(boxes[valid]).all():
            raise ValueError(f"non-finite handbox cache: {path}")
        return boxes, valid

    @staticmethod
    def _validate_keypoints(record: dict, path: Path) -> Tensor:
        if "keypoints" not in record:
            raise KeyError("missing keypoints")
        keypoints = torch.as_tensor(record["keypoints"], dtype=torch.float32)
        if keypoints.shape != (2, 21, 3) or not torch.isfinite(keypoints).all():
            raise ValueError(f"invalid keypoint cache: {path}")
        return keypoints

    def _load(self, path: Path, *, require_keypoints: bool = False) -> dict | None:
        if not path.is_file():
            return None
        try:
            record = torch.load(path, map_location="cpu", weights_only=True)
            boxes, valid = self._validate_boxes(record, path)
            keypoints = self._validate_keypoints(record, path) if require_keypoints else record.get("keypoints")
            if keypoints is not None:
                keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                if keypoints.shape != (2, 21, 3) or not torch.isfinite(keypoints).all():
                    raise ValueError(f"invalid keypoint cache: {path}")
        except (KeyError, RuntimeError, TypeError, ValueError, IndexError):
            if require_keypoints:
                return None
            raise
        return {
            "boxes": boxes,
            "box_valid": valid,
            "valid": valid,
            "keypoints": keypoints,
            "metadata": record.get("metadata", {}),
        }

    def _save(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(record, temporary)
        temporary.replace(path)

    @staticmethod
    def _prediction_record(prediction: dict, *, require_keypoints: bool) -> dict:
        boxes = torch.as_tensor(prediction["boxes"], dtype=torch.float32)
        valid = torch.as_tensor(
            prediction.get("box_valid", prediction.get("valid")), dtype=torch.bool
        )
        if boxes.shape != (2, 4) or valid.shape != (2,):
            raise ValueError("detector returned invalid handbox shapes")
        record = {
            "schema_version": 2 if "keypoints" in prediction else 1,
            "boxes": boxes,
            "box_valid": valid,
            "valid": valid,
            "metadata": prediction.get("metadata", {}),
        }
        if "keypoints" in prediction:
            keypoints = torch.as_tensor(prediction["keypoints"], dtype=torch.float32)
            if keypoints.shape != (2, 21, 3) or not torch.isfinite(keypoints).all():
                raise ValueError("detector returned invalid keypoint shapes")
            record["keypoints"] = keypoints
        elif require_keypoints:
            raise ValueError("detector must return keypoints for get_detections")
        return record

    def _missing_predictions(self, dataset, *, require_keypoints: bool):
        records: list[dict | None] = [None] * len(dataset.frames)
        missing: list[tuple[int, int]] = []
        for ti, frame in enumerate(dataset.frames):
            record = self._load(self._path(dataset.clip_id, frame), require_keypoints=require_keypoints)
            if record is None:
                missing.append((ti, frame))
            else:
                records[ti] = record
        return records, missing

    def _run_missing(self, dataset, records, missing, *, require_keypoints: bool) -> None:
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            images = torch.stack([
                torch.from_numpy(dataset.image(frame)).permute(2, 0, 1)
                for _, frame in chunk
            ])
            predictions = self.detector(images)
            if len(predictions) != len(chunk):
                raise RuntimeError("detector returned an unexpected frame count")
            for (ti, _frame), prediction in zip(chunk, predictions):
                record = self._prediction_record(prediction, require_keypoints=require_keypoints)
                self._save(self._path(dataset.clip_id, dataset.frames[ti]), record)
                records[ti] = record

    def get_boxes(self, dataset) -> tuple[Tensor, Tensor, list[dict]]:
        """Return ``(boxes [2,T,4], valid [2,T], records)``."""
        records, missing = self._missing_predictions(dataset, require_keypoints=False)
        self._run_missing(dataset, records, missing, require_keypoints=False)
        boxes = torch.stack([record["boxes"] for record in records], dim=1)
        valid = torch.stack([record["box_valid"] for record in records], dim=1)
        return boxes, valid, records  # type: ignore[arg-type]

    def get_detections(self, dataset) -> tuple[Tensor, Tensor, Tensor, list[dict]]:
        """Return ``(keypoints, boxes, box_valid, records)`` in frame order."""
        records, missing = self._missing_predictions(dataset, require_keypoints=True)
        self._run_missing(dataset, records, missing, require_keypoints=True)
        keypoints = torch.stack([record["keypoints"] for record in records], dim=1)
        boxes = torch.stack([record["boxes"] for record in records], dim=1)
        valid = torch.stack([record["box_valid"] for record in records], dim=1)
        return keypoints, boxes, valid, records  # type: ignore[arg-type]

    def get_detections_batch(self, datasets) -> list[tuple[Tensor, Tensor, Tensor, list[dict]]]:
        """Process multiple windows with one batched detector stream."""
        records = []
        missing = []
        for dataset_index, dataset in enumerate(datasets):
            frame_records, frame_missing = self._missing_predictions(
                dataset, require_keypoints=True
            )
            records.append(frame_records)
            missing.extend(
                (dataset_index, time_index, frame) for time_index, frame in frame_missing
            )
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            images = torch.stack([
                torch.from_numpy(datasets[dataset_index].image(frame)).permute(2, 0, 1)
                for dataset_index, _time_index, frame in chunk
            ])
            predictions = self.detector(images)
            if len(predictions) != len(chunk):
                raise RuntimeError("detector returned an unexpected frame count")
            for (dataset_index, time_index, frame), prediction in zip(chunk, predictions):
                record = self._prediction_record(prediction, require_keypoints=True)
                dataset = datasets[dataset_index]
                self._save(self._path(dataset.clip_id, frame), record)
                records[dataset_index][time_index] = record
        output = []
        for frame_records in records:
            keypoints = torch.stack([record["keypoints"] for record in frame_records], dim=1)
            boxes = torch.stack([record["boxes"] for record in frame_records], dim=1)
            valid = torch.stack([record["box_valid"] for record in frame_records], dim=1)
            output.append((keypoints, boxes, valid, frame_records))
        return output
