"""Recompute evaluation metrics from saved prediction artifacts.

No model, checkpoint, HaMeR, VGGT, or GPU is needed. This is the second half
of eval/run.py: predictions are loaded once and metric definitions can be
changed/rechecked repeatedly without rerunning inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from src.eval.metric.compute import compute_metrics
from src.eval.metric.camera import camera_metrics
from src.model.flow.state import CameraFlowState, ManoFlowState

_HAND_METRICS = (
    "mpjpe_mm", "mpjpe_scaled_mm", "w_mpjpe_mm", "w_mpjpe_scaled_mm",
    "root_translation_error_mm", "root_translation_error_scaled_mm",
    "root_rotation_error_deg", "w_root_rotation_error_deg",
    "local_rotation_error_deg", "w_local_rotation_error_deg",
    "pa_mpjpe_mm", "wa_mpjpe_mm", "wa_mpjpe_scaled_mm",
    "velocity_error_mmps", "acceleration_error_mmps2", "endpoint_drift_mm",
)


def _mano(payload):
    return ManoFlowState(**payload)


def _camera(payload):
    return CameraFlowState(**payload)


def recompute(predictions: list[dict]) -> tuple[list[dict], dict]:
    records = []
    for artifact in predictions:
        values = dict.fromkeys(_HAND_METRICS)
        prediction_camera = _camera(artifact["prediction_camera"])
        target_camera = _camera(artifact["target_camera"])
        if artifact.get("prediction_joints") is not None:
            prediction_mano = _mano(artifact["prediction_mano"])
            target_mano = _mano(artifact["target_mano"])
            values.update(
                compute_metrics(
                    prediction_mano,
                    target_mano,
                    prediction_camera,
                    target_camera,
                    artifact["prediction_joints"],
                    artifact["target_joints"],
                )
            )
        else:
            values.update(camera_metrics(
                prediction_camera.rotation, target_camera.rotation,
                prediction_camera.translation, target_camera.translation,
            ))
        for name, value in list(values.items()):
            if value is not None:
                value = float(value.detach().cpu())
                values[name] = value if math.isfinite(value) else None
        records.append({
            "clip": artifact["clip"],
            "window": artifact["window"],
            "frame_id_start": artifact["frame_ids"][0],
            "frame_id_end": artifact["frame_ids"][-1],
            **values,
        })

    metrics = {}
    counts = {}
    for name in _HAND_METRICS + (
        "camera_rotation_error_deg", "w_camera_rotation_error_deg",
        "camera_translation_error_mm", "camera_translation_error_scaled_mm",
        "w_camera_translation_error_mm", "w_camera_translation_error_scaled_mm",
        "camera_trajectory_rotation_error_deg", "camera_trajectory_translation_error_mm",
        "camera_trajectory_translation_error_scaled_mm",
    ):
        valid = [record[name] for record in records if record.get(name) is not None]
        counts[name] = len(valid)
        metrics[name] = sum(valid) / len(valid) if len(valid) == len(records) and valid else None
    return records, {"metrics": metrics, "valid_windows_per_metric": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unfiltered", action="store_true", help="Recompute historical unfiltered metrics")
    args = parser.parse_args()
    if not args.unfiltered:
        from src.eval.protocol import evaluate_saved
        print(json.dumps(evaluate_saved(args.predictions, args.output_dir), indent=2, allow_nan=False))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact = torch.load(args.predictions, map_location="cpu", weights_only=True)
    records, summary = recompute(artifact["windows"])
    summary.update({
        "source_predictions": str(args.predictions.resolve()),
        "checkpoint_step": artifact.get("checkpoint_step", -1),
        "windows": len(records),
        "convergence_claim": False,
        "aggregation": "Equal window mean; null if a metric is missing in any window.",
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
