"""Portable paths used by training, evaluation, and external workers.

Every machine-specific path can be overridden with a ``WHG_*`` environment
variable.  Repository-relative third-party paths intentionally remain stable
so the source tree can be moved without editing Python files.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_env(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default).expanduser()


def _command_env(name: str, default: str = "python") -> Path:
    return Path(os.environ.get(name, default)).expanduser()


THIRD_PARTY_ROOT = _path_env("WHG_THIRD_PARTY_ROOT", PROJECT_ROOT / "third_party")

# Read-only external data mounts. Set these before training or evaluation.
DATA_ROOT = _path_env("WHG_DATA_ROOT", PROJECT_ROOT / "data" / "hot3dclip")
EGODEX_ROOT = _path_env("WHG_EGODEX_ROOT", PROJECT_ROOT / "data" / "egodex")
HOT3D_METRIC_ROOT = _path_env("WHG_HOT3D_METRIC_ROOT", THIRD_PARTY_ROOT / "hot3d-metric")

# Runtime-generated files use an ignored project-local directory by default.
# Set WHG_CACHE_ROOT and WHG_EXPERIMENT_ROOT to place them elsewhere.
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"
CACHE_ROOT = _path_env("WHG_CACHE_ROOT", RUNTIME_ROOT / "cache")
EXPERIMENT_ROOT = _path_env("WHG_EXPERIMENT_ROOT", RUNTIME_ROOT / "experiments" / "default")

MANO_ROOT = _path_env("WHG_MANO_ROOT", THIRD_PARTY_ROOT / "mano" / "assets")
HAMER_REPOSITORY = _path_env("WHG_HAMER_REPOSITORY", THIRD_PARTY_ROOT / "hamer")
HAMER_CHECKPOINT = _path_env(
    "WHG_HAMER_CHECKPOINT",
    HAMER_REPOSITORY / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
)
VGGT_REPOSITORY = _path_env("WHG_VGGT_REPOSITORY", THIRD_PARTY_ROOT / "vggt-omega")
VGGT_CHECKPOINT = _path_env(
    "WHG_VGGT_CHECKPOINT", VGGT_REPOSITORY / "checkpoints" / "vggt_omega_1b_512.pt"
)
DETECTOR_CHECKPOINT = _path_env(
    "WHG_DETECTOR_CHECKPOINT",
    HAMER_REPOSITORY / "_DATA" / "detectron2" / "model_final_f05665.pkl",
)

# External workers may use different environments. The defaults are portable
# command names; setup scripts print exact interpreter paths after installation.
HAMER_PYTHON = _command_env("WHG_HAMER_PYTHON")
DETECTOR_PYTHON = _command_env("WHG_DETECTOR_PYTHON")
VGGT_PYTHON = _command_env("WHG_VGGT_PYTHON")

# Short aliases for callers that refer to repositories as roots.
HAMER_ROOT = HAMER_REPOSITORY
VGGT_ROOT = VGGT_REPOSITORY
