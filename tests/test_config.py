from __future__ import annotations

import importlib
from pathlib import Path


def test_config_defaults_are_portable(monkeypatch) -> None:
    for name in list(vars(__import__("os")).get("environ", {})):
        if name.startswith("WHG_"):
            monkeypatch.delenv(name, raising=False)
    import src.config as config

    config = importlib.reload(config)
    assert config.PROJECT_ROOT == Path(config.__file__).resolve().parents[1]
    assert config.DATA_ROOT == config.PROJECT_ROOT / "data" / "hot3dclip"
    assert config.CACHE_ROOT == config.PROJECT_ROOT / ".runtime" / "cache"
    assert config.EXPERIMENT_ROOT == config.PROJECT_ROOT / ".runtime" / "experiments" / "default"
    assert config.HAMER_PYTHON == Path("python")
    assert config.DETECTOR_PYTHON == Path("python")
    assert config.VGGT_PYTHON == Path("python")
    assert config.HAMER_REPOSITORY == config.PROJECT_ROOT / "third_party" / "hamer"
    assert config.VGGT_CHECKPOINT == config.VGGT_REPOSITORY / "checkpoints" / "vggt_omega_1b_512.pt"


def test_config_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    overrides = {
        "WHG_DATA_ROOT": tmp_path / "hot3d",
        "WHG_CACHE_ROOT": tmp_path / "cache",
        "WHG_EXPERIMENT_ROOT": tmp_path / "experiments",
        "WHG_HAMER_PYTHON": tmp_path / "hamer-python",
        "WHG_DETECTOR_CHECKPOINT": tmp_path / "detector.pkl",
        "WHG_THIRD_PARTY_ROOT": tmp_path / "third-party",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, str(value))
    import src.config as config

    config = importlib.reload(config)
    assert config.DATA_ROOT == overrides["WHG_DATA_ROOT"]
    assert config.CACHE_ROOT == overrides["WHG_CACHE_ROOT"]
    assert config.EXPERIMENT_ROOT == overrides["WHG_EXPERIMENT_ROOT"]
    assert config.HAMER_PYTHON == overrides["WHG_HAMER_PYTHON"]
    assert config.DETECTOR_CHECKPOINT == overrides["WHG_DETECTOR_CHECKPOINT"]
    assert config.THIRD_PARTY_ROOT == overrides["WHG_THIRD_PARTY_ROOT"]
