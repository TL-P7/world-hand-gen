"""Pre-training HOT3D train.Aria continuity validation.

This command is intentionally separate from training: it inspects directories
and frame numbering once, before launching a run.
"""

from __future__ import annotations

import argparse
import json

from src.config import DATA_ROOT
from src.data.hot3d import frame_ids
from src.data.hot3d.splits import partition_clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args()

    official = json.loads((DATA_ROOT / "clip_splits.json").read_text())
    mounted = [f"{clip_id:06d}" for clip_id in official["train"]["Aria"]]
    train, test = partition_clips(mounted, train_count=1272, seed=2026)
    clips = train if args.split == "train" else test
    total_frames = 0
    short = []
    missing = []
    for clip in clips:
        ids = frame_ids(clip)
        total_frames += len(ids)
        if len(ids) < 64:
            short.append(clip)
        if any(right != left + 1 for left, right in zip(ids, ids[1:])):
            missing.append(clip)
    print(f"{args.split}.Aria clips={len(clips)} frames={total_frames}")
    print(f"shorter_than_64={len(short)} non_contiguous={len(missing)}")
    if short:
        print("short clips:", ",".join(short))
    if missing:
        print("non-contiguous clips:", ",".join(missing))


if __name__ == "__main__":
    main()
