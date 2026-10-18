"""Deterministic local train/test partition for the mounted train_aria clips."""

from __future__ import annotations

import random


def partition_clips(
    clip_ids: list[str], *, train_count: int = 1272, seed: int = 2026
) -> tuple[list[str], list[str]]:
    """Split sorted clip ids into deterministic train and test subsets."""
    names = sorted(clip_ids)
    if len(names) != len(set(names)):
        raise ValueError("clip_ids contains duplicates")
    if not 0 < train_count < len(names):
        raise ValueError("train_count must lie in (0,len(clip_ids))")
    shuffled = names.copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count:]
