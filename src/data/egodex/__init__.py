"""Minimal EgoDex RGB and OP-16 world-coordinate hand-joint access.

Requires h5py and OpenCV. Joint coordinates retain the dataset's raw units;
no camera transform, image rotation, projection, or MANO fitting is applied.
"""

from __future__ import annotations

from pathlib import Path

from src.config import EGODEX_ROOT

from .dataset import EgoDexDataset, JOINT_SUFFIXES

__all__ = ["EgoDexDataset", "JOINT_SUFFIXES", "episode_ids"]


def episode_ids(split: str = "train", root: Path | None = None) -> list[str]:
    """Return paired episode IDs relative to the root, using official splits."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root = EGODEX_ROOT if root is None else Path(root)
    parts = [f"part{number}" for number in range(1, 6)] if split == "train" else ["test"]
    episodes = []
    for part in parts:
        directory = root / part
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for task in sorted(directory.iterdir()):
            if not task.is_dir():
                continue
            annotations = {path.stem for path in task.glob("*.hdf5")}
            videos = {path.stem for path in task.glob("*.mp4")}
            if annotations != videos:
                missing = sorted(annotations ^ videos)[0]
                raise ValueError(f"unpaired EgoDex episode: {task / missing}")
            episodes.extend(f"{part}/{task.name}/{stem}" for stem in sorted(annotations))
    return episodes
