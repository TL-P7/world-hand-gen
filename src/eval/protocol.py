"""Evaluate saved HOT3D predictions with the standalone GT visibility protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

from src.config import CACHE_ROOT, DATA_ROOT, HOT3D_METRIC_ROOT


PROTOCOL = "hot3d-metric/gt-amodal-upright-384/hand-offscreen-ge-32"


def _metric():
    if not HOT3D_METRIC_ROOT.is_dir():
        raise FileNotFoundError(f"missing HOT3D metric repository: {HOT3D_METRIC_ROOT}")
    if str(HOT3D_METRIC_ROOT) not in sys.path:
        sys.path.insert(0, str(HOT3D_METRIC_ROOT))
    from hot3d_metric.evaluate import evaluate
    from hot3d_metric.recompute import CAMERA_METRICS, HAND_METRICS
    from hot3d_metric.visibility import offscreen_from_hot3d
    return evaluate, HAND_METRICS, CAMERA_METRICS, offscreen_from_hot3d


def _key(window):
    return window["clip"], int(window["window"]), tuple(window["frame_ids"])


def _visibility(windows, *, cache_root, data_root, output_size, offscreen_from_hot3d):
    data_root = Path(data_root).resolve()
    cache_root = Path(cache_root) / "hot3d-metric-visibility-v1" / f"{output_size}"
    entries = {}
    for window in windows:
        key = _key(window)
        if key in entries or len(key[2]) != 64 or key[1] not in (0, 1):
            raise ValueError(f"duplicate or invalid 64-frame window: {key}")
        frames = list(key[2])
        signature = hashlib.sha256(json.dumps([str(data_root), key, output_size]).encode()).hexdigest()[:16]
        path = cache_root / key[0] / f"{key[1]}_{signature}.json"
        if path.is_file():
            cached = json.loads(path.read_text())
            if cached["key"] != [key[0], key[1], frames] or cached["data_root"] != str(data_root):
                raise ValueError(f"visibility cache key mismatch: {path}")
            masks = cached["offscreen"]
        else:
            masks = offscreen_from_hot3d(data_root, key[0], frames, output_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "key": [key[0], key[1], frames], "data_root": str(data_root),
                "output_size": output_size, "protocol": PROTOCOL, "offscreen": masks,
            }) + "\n")
            temporary.replace(path)
        if set(masks) != {"left", "right"} or any(
            not isinstance(masks[hand], list) or len(masks[hand]) != 64
            or any(type(value) is not bool for value in masks[hand])
            for hand in ("left", "right")
        ):
            raise ValueError(f"invalid visibility masks: {path}")
        entries[key] = masks
    return entries


def evaluate_saved(predictions: Path, output_dir: Path, *, cache_root=CACHE_ROOT,
                   data_root=DATA_ROOT, output_size=384, expected_windows=None):
    """Write protocol summaries without modifying inference artifacts or existing outputs."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    evaluate, hand_metrics, camera_metrics, offscreen_from_hot3d = _metric()
    source = torch.load(predictions, map_location="cpu", weights_only=True)
    windows = source["windows"]
    if not windows or (expected_windows is not None and len(windows) != expected_windows):
        raise ValueError(f"expected {expected_windows} windows, got {len(windows)}")
    masks = _visibility(windows, cache_root=cache_root, data_root=data_root,
                        output_size=output_size, offscreen_from_hot3d=offscreen_from_hot3d)
    keyed = [{**window, "offscreen": masks[_key(window)]} for window in windows]
    records, summary = evaluate(keyed, filter_offscreen=True, output_size=output_size)
    if len(records) != len(windows) or len(summary["metrics"]) != len(hand_metrics + camera_metrics):
        raise ValueError("HOT3D metric returned incomplete records")
    summary.update({
        "source_predictions": str(predictions.resolve()), "source_targets": None,
        "checkpoint_step": source.get("checkpoint_step", -1),
        "split": "test244" if len(windows) == 488 else "test244-prefix",
        "clips": len({window["clip"] for window in windows}),
        "smoke_only": len(windows) != 488,
        "aggregation": "Equal window mean: hand metrics use hand_windows, camera metrics use all windows; null if any eligible window lacks that metric.",
        "convergence_claim": False, "protocol": PROTOCOL,
        "visibility_source": "GT boxes_amodal 214-1 rectified to upright 384x384",
        "data_root": str(Path(data_root).resolve()), "output_size": output_size,
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "records.csv").open("w", newline="") as handle:
        fields = ["clip", "window", "frame_id_start", "frame_id_end", "hand_excluded",
                  "left_offscreen_frames", "right_offscreen_frames", *hand_metrics, *camera_metrics]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "visibility_masks.json").write_text(json.dumps({
        "protocol": PROTOCOL,
        "windows": [{"clip": key[0], "window": key[1], "frame_ids": list(key[2]),
                     "offscreen": masks[key]} for key in masks],
    }) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--expected-windows", type=int)
    args = parser.parse_args()
    result = evaluate_saved(args.predictions, args.output_dir, cache_root=args.cache_root,
                            data_root=args.data_root, expected_windows=args.expected_windows)
    print(json.dumps({key: result[key] for key in ("checkpoint_step", "windows", "hand_windows", "excluded_windows")},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
