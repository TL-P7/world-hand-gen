import pytest
import sys
import torch
from pathlib import Path

import src.eval.run as run
from src.model.flow.state import CameraFlowState, ManoFlowState


def test_main_skips_vggt_worker_only_when_all_windows_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(run, "_test_clips", lambda: ["001849"])
    monkeypatch.setattr(run, "frame_ids", lambda clip: list(range(150)))
    monkeypatch.setattr(run, "_load_checkpoint", lambda *args: (12000, False))
    monkeypatch.setattr(run, "MMDiT", lambda **kwargs: _EvalModule())
    monkeypatch.setattr(run, "ConditionEncoder", lambda **kwargs: _EvalModule())
    monkeypatch.setattr(run, "ManoWrapper", _EvalModule)
    monkeypatch.setattr(run, "Hot3DRectifier", lambda **kwargs: object())
    monkeypatch.setattr(run, "HandBoxCache", lambda **kwargs: object())
    monkeypatch.setattr(run, "HaMeRTokenCache", lambda **kwargs: object())
    workers = []
    used_caches = []

    class Worker:
        def __init__(self, _python, kind, _repository, _checkpoint, **kwargs):
            self.kind = kind
            self.closed = False
            workers.append(self)

        def close(self):
            self.closed = True

    def fake_evaluate(**kwargs):
        used_caches.append(kwargs["vggt_cache"])
        kwargs["prediction_artifact"].append({"clip": "001849"})
        return [{"missing_mano_count": 0, "nonfinite_metrics": "", "metric": 1.0}]

    monkeypatch.setattr(run, "DetectorWorker", lambda *args, **kwargs: Worker(None, "detector", None, None))
    monkeypatch.setattr(run, "FeatureWorker", Worker)
    monkeypatch.setattr(run, "evaluate", fake_evaluate)
    def fake_protocol(predictions, output, **kwargs):
        output.mkdir()
        (output / "records.csv").write_text("clip,window\n001849,0\n")
        result = {"protocol": "test", "metrics": {"metric": 1.0}}
        (output / "summary.json").write_text('{"protocol":"test"}')
        return result

    monkeypatch.setattr(run, "evaluate_saved", fake_protocol)
    cache = run.VGGTOmegaTokenCache(cache_root=tmp_path / "cache", vggt=None)
    for frames in (list(range(64)), list(range(64, 128))):
        path = cache._path("001849", frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def invoke(output):
        monkeypatch.setattr(sys, "argv", ["eval", "--checkpoint", str(tmp_path / "checkpoint.pt"),
                                        "--output-dir", str(output), "--device", "cpu"])
        run.main()

    invoke(tmp_path / "complete")
    assert [worker.kind for worker in workers] == ["detector", "hamer"]
    assert used_caches[-1].vggt is None
    assert all(worker.closed for worker in workers)
    for name in ("request.json", "summary.json", "predictions.pt", "records.csv"):
        assert (tmp_path / "complete" / name).is_file()
    with pytest.raises(FileExistsError):
        invoke(tmp_path / "complete")
    assert len(workers) == 2

    cache._path("001849", list(range(64, 128))).unlink()
    invoke(tmp_path / "partial")
    assert [worker.kind for worker in workers[2:]] == ["detector", "hamer", "vggt"]
    assert used_caches[-1].vggt is workers[-1]
    assert all(worker.closed for worker in workers)


class _EvalModule:
    def to(self, device):
        return self

    def eval(self):
        return self


class _Dataset:
    def __init__(self, clip, frames, rectifier, **kwargs):
        self.clip_id, self.frames = clip, frames

    def image(self, frame):
        return frame

    def hands(self, frame):
        # Exercise the explicit missing-target semantics without invoking MANO.
        return {"left": {}, "right": {}}

    def camera_pose(self, frame):
        return {}

    def hand_box(self, hand, frame):
        raise AssertionError("evaluation must not use GroundTruth hand boxes")

    def targets(self, rectifier):
        return (
            ManoFlowState(torch.zeros(1, 2, 64, 3), torch.zeros(1, 2, 64, 6), torch.zeros(1, 2, 64, 15, 6)),
            CameraFlowState(torch.zeros(1, 64, 6), torch.zeros(1, 64, 3)),
        )


class _HandBoxCache:
    def __init__(self):
        self.cache_root = Path("/tmp")
        self.calls = []

    def get_boxes(self, dataset):
        self.calls.append(tuple(dataset.frames))
        return torch.zeros(2, 64, 4), torch.ones(2, 64, dtype=torch.bool), []


class _Cache:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get_tokens(self, dataset, boxes=None, valid=None):
        self.calls.append(tuple(dataset.frames))
        if self.value == "hamer":
            return torch.zeros(2, 64, 4), torch.ones(2, 64, dtype=torch.bool)
        return torch.zeros(1, 64, 4)


class _Encoder:
    def eval(self):
        return self

    def __call__(self, *args):
        return object()


class _Model:
    def eval(self):
        return self


class _Mano:
    def eval(self):
        return self


def test_windows_are_exact_fixed_zero_based_ranges(monkeypatch):
    monkeypatch.setattr(run, "frame_ids", lambda clip: list(range(100, 250)))
    assert run._windows("000001") == [list(range(100, 164)), list(range(164, 228))]


def test_evaluate_two_windows_and_seed_reproducible(monkeypatch):
    monkeypatch.setattr(run, "Hot3DDataset", _Dataset)
    monkeypatch.setattr(run, "_windows", lambda clip: [list(range(10, 74)), list(range(74, 138))])
    monkeypatch.setattr(run, "sample_noise_state", lambda batch, time, device, generator: (generator.initial_seed(), generator.initial_seed()))
    seen = []

    def fake_sample(model, conditions, *, batch, time, steps, device, noise):
        seen.append(noise[0])
        return (
            ManoFlowState(torch.zeros(1, 2, 64, 3), torch.zeros(1, 2, 64, 6), torch.zeros(1, 2, 64, 15, 6)),
            CameraFlowState(torch.zeros(1, 64, 6), torch.zeros(1, 64, 3)),
        )

    monkeypatch.setattr(run, "sample", fake_sample)
    monkeypatch.setattr(run, "camera_metrics", lambda *args: {"camera_metric": torch.tensor(1.)})
    hamer, vggt, handbox = _Cache("hamer"), _Cache("vggt"), _HandBoxCache()
    kwargs = dict(clips=["000010"], model=_Model(), condition_encoder=_Encoder(), mano=_Mano(),
                  handbox_cache=handbox, hamer_cache=hamer, vggt_cache=vggt, rectifier=None, device="cpu", seed=7)
    first = run.evaluate(**kwargs)
    second = run.evaluate(**kwargs)
    assert len(first) == len(second) == 2
    assert [row["window"] for row in first] == [0, 1]
    assert [row["frame_index_start"] for row in first] == [0, 64]
    assert [row["frame_display_start"] for row in first] == [1, 65]
    assert [row["seed"] for row in first] == [row["seed"] for row in second]
    assert seen == [seen[0], seen[1], seen[0], seen[1]]
    assert len(hamer.calls) == len(vggt.calls) == 4
