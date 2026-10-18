"""Fixed-window local test244 evaluation (not evidence of training convergence)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from src.config import (
    CACHE_ROOT,
    DATA_ROOT,
    DETECTOR_CHECKPOINT,
    DETECTOR_PYTHON,
    HAMER_CHECKPOINT,
    HAMER_PYTHON,
    HAMER_REPOSITORY,
    VGGT_CHECKPOINT,
    VGGT_PYTHON,
    VGGT_REPOSITORY,
)
from src.data.hot3d import Hot3DRectifier, frame_ids
from src.data.hot3d.cache import HaMeRTokenCache, VGGTOmegaTokenCache
from src.data.hot3d.handbox import HandBoxCache
from src.data.hot3d.dataset import Hot3DDataset
from src.data.hot3d.splits import partition_clips
from src.eval.metric.aligned import pa_mpjpe_mm, wa_mpjpe_mm, wa_mpjpe_scaled_mm
from src.eval.metric.camera import camera_metrics
from src.eval.metric.compute import compute_metrics
from src.eval.protocol import evaluate_saved
from src.model.flow.condition import ConditionEncoder
from src.model.flow.mmdit import MMDiT
from src.model.flow.sampler import sample, sample_noise_state
from src.model.wrappers import ManoWrapper
from src.train.feature_worker import FeatureWorker
from src.train.detector_worker import DetectorWorker

_HAND_METRICS = (
    "mpjpe_mm", "mpjpe_scaled_mm", "w_mpjpe_mm", "w_mpjpe_scaled_mm",
    "root_translation_error_mm", "root_translation_error_scaled_mm",
    "root_rotation_error_deg", "w_root_rotation_error_deg", "local_rotation_error_deg",
    "w_local_rotation_error_deg", "pa_mpjpe_mm", "wa_mpjpe_mm", "wa_mpjpe_scaled_mm",
    "velocity_error_mmps", "acceleration_error_mmps2", "endpoint_drift_mm",
)


def _test_clips() -> list[str]:
    official = json.loads((DATA_ROOT / "clip_splits.json").read_text())
    mounted = [f"{x:06d}" for x in official["train"]["Aria"]]
    test = partition_clips(mounted, train_count=1272, seed=2026)[1]
    if len(test) != 244:
        raise ValueError(f"expected local test244 split, got {len(test)} clips")
    return test


def _windows(clip: str) -> list[list[int]]:
    frames = frame_ids(clip)
    if len(frames) != 150 or frames != list(range(frames[0], frames[0] + 150)):
        raise ValueError(f"expected 150 contiguous frames for clip {clip}")
    return [frames[:64], frames[64:128]]


def _load_checkpoint(path, model, condition_encoder):
    # Optimizer/RNG state stays on CPU and is discarded, never copied to the GPU.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    condition_encoder.load_state_dict(payload["condition_encoder"])
    return int(payload.get("step", -1)), bool(payload.get("smoke_only", False))


@torch.no_grad()
def evaluate(*, clips, model, condition_encoder, mano, handbox_cache, hamer_cache, vggt_cache,
             rectifier, device, seed=0, checkpoint_step=-1, smoke_only=False,
             prediction_artifact=None):
    """Return one record per fixed window, using injectable models and caches.

    Missing MANO annotations invalidate all hand metrics for that window: these
    metrics lack validity-mask support. Keep the window and camera metrics, and
    report missing hand metrics as None, never as zero or a filtered subset.
    """
    device = torch.device(device)
    model.eval()
    condition_encoder.eval()
    mano.eval()
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for clip_index, clip in enumerate(clips):
            for window_index, frames in enumerate(_windows(clip)):
                dataset = Hot3DDataset(
                    clip, frames, rectifier,
                    image_cache_root=handbox_cache.cache_root / "rectified-v1",
                )
                if hasattr(handbox_cache, "get_detections"):
                    keypoints_2d, boxes, box_valid, _ = handbox_cache.get_detections(dataset)
                else:
                    boxes, box_valid, detection_records = handbox_cache.get_boxes(dataset)
                    if detection_records and all(record is not None for record in detection_records):
                        keypoints_2d = torch.stack(
                            [torch.as_tensor(record.get("keypoints", torch.zeros(2, 21, 3))) for record in detection_records],
                            dim=1,
                        )
                    else:
                        keypoints_2d = torch.zeros(2, len(dataset.frames), 21, 3)
                bbox = torch.zeros(1, 2, 64, 4)
                bbox_valid = torch.zeros(1, 2, 64, dtype=torch.bool)
                bbox[0] = boxes / getattr(rectifier, "output_size", 384)
                bbox_valid[0] = box_valid
                keypoints = keypoints_2d.unsqueeze(0).float()
                keypoints[..., :2] = keypoints[..., :2] / getattr(rectifier, "output_size", 384)
                missing = []
                # Warm every shared lazy input before concurrent cache access.
                for ti, frame in enumerate(frames):
                    dataset.image(frame)
                    hands = dataset.hands(frame)
                    dataset.camera_pose(frame)
                    for hi, hand in enumerate(("left", "right")):
                        if hands.get(hand, {}).get("mano_pose") is None:
                            missing.append(f"{hand}@{frame}")
                if missing:
                    warnings.warn(
                        f"clip {clip} window {window_index} frames {frames[0]}..{frames[-1]}: "
                        f"{len(missing)}/128 missing MANO targets ({', '.join(missing[:4])}); "
                        "hand metrics are null, window retained; see records.csv for all missing targets",
                        stacklevel=2,
                    )
                # no_grad is thread-local; explicitly wrap both worker tasks.
                hamer_future = pool.submit(
                    torch.no_grad()(hamer_cache.get_tokens), dataset, boxes, box_valid
                )
                vggt_future = pool.submit(torch.no_grad()(vggt_cache.get_tokens), dataset)
                hamer_tokens, hamer_valid = hamer_future.result()
                vggt_tokens = vggt_future.result()
                target_mano, target_camera = dataset.targets(rectifier)
                target_mano = target_mano.map(lambda value: value.to(device))
                target_camera = target_camera.map(lambda value: value.to(device))
                conditions = condition_encoder(
                    hamer_tokens.unsqueeze(0).to(device), vggt_tokens.to(device), bbox.to(device),
                    hamer_valid.unsqueeze(0).to(device),
                    torch.ones(1, 64, dtype=torch.bool, device=device), bbox_valid.to(device),
                    keypoints.to(device),
                )
                # Independent of processing order, truncation, and global RNG consumption.
                window_seed = (seed + clip_index * 2 + window_index) % (2**63)
                noise = sample_noise_state(
                    1, 64, device, generator=torch.Generator(device=device).manual_seed(window_seed),
                )
                pred_mano, pred_camera = sample(
                    model, conditions, batch=1, time=64, steps=10, device=device, noise=noise,
                )
                values = dict.fromkeys(_HAND_METRICS)
                pred_joints = target_joints = None
                if not missing:
                    pred_joints = mano(pred_mano)
                    target_joints = mano(target_mano)
                    values.update(
                        compute_metrics(
                            pred_mano, target_mano, pred_camera, target_camera,
                            pred_joints, target_joints,
                        )
                    )
                if missing:
                    values.update(camera_metrics(
                        pred_camera.rotation, target_camera.rotation,
                        pred_camera.translation, target_camera.translation,
                    ))
                if prediction_artifact is not None:
                    prediction_artifact.append({
                        "clip": clip,
                        "window": window_index,
                        "frame_ids": list(frames),
                        "prediction_mano": {
                            name: getattr(pred_mano, name).detach().cpu()
                            for name in pred_mano.__dataclass_fields__
                        },
                        "target_mano": {
                            name: getattr(target_mano, name).detach().cpu()
                            for name in target_mano.__dataclass_fields__
                        },
                        "prediction_camera": {
                            name: getattr(pred_camera, name).detach().cpu()
                            for name in pred_camera.__dataclass_fields__
                        },
                        "target_camera": {
                            name: getattr(target_camera, name).detach().cpu()
                            for name in target_camera.__dataclass_fields__
                        },
                        "prediction_joints": None if pred_joints is None else pred_joints.detach().cpu(),
                        "target_joints": None if target_joints is None else target_joints.detach().cpu(),
                        "missing_mano": list(missing),
                    })
                nonfinite = []
                for name, value in values.items():
                    if value is not None:
                        value = float(value.detach().cpu())
                        if not math.isfinite(value):
                            nonfinite.append(name)
                            value = None
                        values[name] = value
                if nonfinite:
                    warnings.warn(f"clip {clip} window {window_index}: nonfinite metrics {nonfinite}; reported as null", stacklevel=2)
                records.append({
                    "clip": clip, "clip_index": clip_index, "clip_display": clip_index + 1,
                    "window": window_index, "window_display": window_index + 1,
                    "frame_index_start": window_index * 64, "frame_index_end": window_index * 64 + 63,
                    "frame_display_start": window_index * 64 + 1, "frame_display_end": window_index * 64 + 64,
                    "frame_id_start": frames[0], "frame_id_end": frames[-1],
                    "seed": window_seed, "checkpoint_step": checkpoint_step, "smoke_only": smoke_only,
                    "missing_mano_count": len(missing), "missing_mano": ";".join(missing),
                    "nonfinite_metrics": ";".join(nonfinite), **values,
                })
    return records


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=None, help="Positive prefix limit; 1 runs a two-window smoke evaluation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hamer-device", default=None)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--vggt-device", default=None)
    args = parser.parse_args()
    args.cache_root = CACHE_ROOT
    if args.hamer_device is None:
        args.hamer_device = "cuda:2" if str(args.device) == "cuda:1" else str(args.device)
    if args.detector_device is None:
        args.detector_device = args.hamer_device
    if args.vggt_device is None:
        args.vggt_device = "cuda:3" if str(args.device) == "cuda:1" else str(args.device)
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")
    clips = _test_clips()[:args.max_clips]
    vggt_cache = VGGTOmegaTokenCache(cache_root=args.cache_root, vggt=None)
    vggt_cached = all(
        vggt_cache._path(clip, frames).is_file()
        for clip in clips for frames in _windows(clip)
    )
    # Reserve before model/worker construction; even an empty existing directory is refused.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = MMDiT(model_dim=512, depth=8, heads=8)
    condition_encoder = ConditionEncoder(model_dim=512)
    step, checkpoint_smoke = _load_checkpoint(args.checkpoint, model, condition_encoder)
    smoke_only = checkpoint_smoke or args.max_clips is not None
    request = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": step,
        "checkpoint_smoke_only": checkpoint_smoke, "smoke_only": smoke_only,
        "output_dir": str(args.output_dir.resolve()), "cache_root": str(args.cache_root.resolve()),
        "device": str(device), "seed": args.seed, "seed_rule": "(seed + 2 * clip_index + window_index) % 2**63",
        "split": "test244", "split_seed": 2026, "train_count": 1272, "test_count": 244,
        "clips": clips, "max_clips": args.max_clips,
        "frame_index_ranges": [[0, 63], [64, 127]], "frame_display_ranges": [[1, 64], [65, 128]],
        "range_end_inclusive": True, "model": {"model_dim": 512, "depth": 8, "heads": 8},
        "sampling_steps": 10, "hamer_python": str(HAMER_PYTHON), "detector_python": str(DETECTOR_PYTHON), "vggt_python": str(VGGT_PYTHON),
        "hamer_device": args.hamer_device, "vggt_device": args.vggt_device,
        "missing_policy": "Any missing MANO target invalidates all hand metrics for that window; no windows dropped",
        "convergence_claim": False,
    }
    (args.output_dir / "request.json").write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
    model.to(device).eval()
    condition_encoder.to(device).eval()
    mano = ManoWrapper().to(device).eval()
    rectifier = Hot3DRectifier(output_size=384)
    detector = None
    hamer = None
    vggt = None
    artifacts = []
    try:
        detector = DetectorWorker(
            DETECTOR_PYTHON, HAMER_REPOSITORY,
            DETECTOR_CHECKPOINT,
            device=str(args.hamer_device), image_size=384, precision="fp16",
        )
        hamer = FeatureWorker(HAMER_PYTHON, "hamer", HAMER_REPOSITORY,
                              HAMER_CHECKPOINT, device=str(args.hamer_device))
        if not vggt_cached:
            vggt = FeatureWorker(VGGT_PYTHON, "vggt", VGGT_REPOSITORY,
                                 VGGT_CHECKPOINT, device=str(args.vggt_device))
        vggt_cache.vggt = vggt
        records = evaluate(
            clips=clips, model=model, condition_encoder=condition_encoder, mano=mano,
            handbox_cache=HandBoxCache(cache_root=args.cache_root, detector=detector),
            hamer_cache=HaMeRTokenCache(cache_root=args.cache_root, hamer=hamer),
            vggt_cache=vggt_cache,
            rectifier=rectifier, device=device, seed=args.seed, checkpoint_step=step,
            smoke_only=smoke_only, prediction_artifact=artifacts,
        )
    finally:
        try:
            if vggt is not None:
                vggt.close()
        finally:
            try:
                if hamer is not None:
                    hamer.close()
            finally:
                if detector is not None:
                    detector.close()

    temporary_predictions = args.output_dir / "predictions.tmp"
    torch.save({"version": 1, "checkpoint_step": step, "windows": artifacts}, temporary_predictions)
    temporary_predictions.replace(args.output_dir / "predictions.pt")

    with (args.output_dir / "unfiltered_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    # Do not silently average a subset when a requested window is unscorable.
    metric_names = list(records[0])[list(records[0]).index("nonfinite_metrics") + 1:]
    metrics, counts = {}, {}
    for name in metric_names:
        valid = [r[name] for r in records if r[name] is not None]
        counts[name] = len(valid)
        mean = math.fsum(v / len(records) for v in valid) if len(valid) == len(records) else None
        metrics[name] = mean if mean is not None and math.isfinite(mean) else None
    summary = {
        "checkpoint_step": step, "seed": args.seed, "smoke_only": smoke_only,
        "convergence_claim": False, "split": "test244", "clips": len(clips), "windows": len(records),
        "missing_mano_windows": sum(r["missing_mano_count"] > 0 for r in records),
        "missing_mano_count": sum(r["missing_mano_count"] for r in records),
        "nonfinite_metric_windows": sum(bool(r["nonfinite_metrics"]) for r in records),
        "metrics": metrics, "valid_windows_per_metric": counts,
        "aggregation": "Equal window mean; null if any requested window is missing/nonfinite. CSV empty metric cells mean null.",
    }
    serialized = json.dumps(summary, indent=2, allow_nan=False)
    (args.output_dir / "unfiltered_summary.json").write_text(serialized + "\n")
    summary = evaluate_saved(args.output_dir / "predictions.pt", args.output_dir / "protocol",
                             expected_windows=len(clips) * 2)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "records.csv").write_bytes((args.output_dir / "protocol" / "records.csv").read_bytes())
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
