from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from src.model.flow.state import CameraFlowState, ManoFlowState
from src.train import train


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_checkpoint_roundtrip_restores_parameters_optimizer_and_rng(
    tmp_path, monkeypatch, device_name
) -> None:
    device = torch.device(device_name)
    cuda_devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1234)
        model = nn.Linear(3, 2).to(device)
        encoder = nn.Linear(2, 3).to(device)
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *encoder.parameters()], lr=0.02, betas=(0.8, 0.95)
        )
        loss = model(encoder(torch.ones(4, 2, device=device))).square().mean()
        loss.backward()
        optimizer.step()
        assert len(optimizer.state) == 4

        path = tmp_path / "checkpoint.pt"
        payload = train._checkpoint_payload(40000, model, encoder, optimizer, seed=1234)
        train._save_checkpoint(path, payload)
        assert path.is_file()
        assert not path.with_suffix(".tmp").exists()
        expected_cpu_rng = torch.get_rng_state().clone()
        expected_cpu_draw = torch.rand(8)
        expected_cuda_rng = torch.cuda.get_rng_state_all() if cuda_devices else []
        expected_cuda_draw = torch.rand(8, device=device) if cuda_devices else None

        restored_model = nn.Linear(3, 2).to(device)
        restored_encoder = nn.Linear(2, 3).to(device)
        restored_optimizer = torch.optim.AdamW(
            [*restored_model.parameters(), *restored_encoder.parameters()], lr=0.9
        )
        torch.rand(17)
        load = Mock(wraps=torch.load)
        monkeypatch.setattr(torch, "load", load)
        step = train._load_checkpoint(
            path, restored_model, restored_encoder, restored_optimizer, device
        )

        assert step == 40000
        assert load.call_args.kwargs["map_location"] == "cpu"
        assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
        assert torch.equal(torch.rand(8), expected_cpu_draw)
        for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda_devices else [], expected_cuda_rng):
            assert torch.equal(actual, expected)
        if cuda_devices:
            assert torch.equal(torch.rand(8, device=device), expected_cuda_draw)

        for restored, original in [(restored_model, model), (restored_encoder, encoder)]:
            for name, value in original.state_dict().items():
                assert torch.equal(restored.state_dict()[name], value)
        expected_optimizer = optimizer.state_dict()
        actual_optimizer = restored_optimizer.state_dict()
        assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
        assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
        for parameter_id, state in expected_optimizer["state"].items():
            for name, value in state.items():
                assert torch.equal(actual_optimizer["state"][parameter_id][name].cpu(), value.cpu())
        for parameter, state in restored_optimizer.state.items():
            # Non-capturable AdamW keeps its scalar step on CPU, but moments
            # must follow the parameter device when loading a CPU checkpoint.
            assert state["exp_avg"].device == parameter.device == device
            assert state["exp_avg_sq"].device == parameter.device

        restored_optimizer.zero_grad(set_to_none=True)
        restored_model(restored_encoder(torch.ones(4, 2, device=device))).sum().backward()
        restored_optimizer.step()
        assert all(state["step"].item() == 2 for state in restored_optimizer.state.values())


@pytest.mark.parametrize("saved_device_count", [0, 1, 3])
def test_load_checkpoint_accepts_different_cuda_rng_device_counts(
    tmp_path, monkeypatch, saved_device_count
) -> None:
    model = nn.Linear(2, 2)
    encoder = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([*model.parameters(), *encoder.parameters()])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    payload = train._checkpoint_payload(40000, model, encoder, optimizer, seed=0)
    saved_states = [torch.tensor([index], dtype=torch.uint8) for index in range(saved_device_count)]
    payload["cuda_rng_state"] = saved_states
    path = tmp_path / "checkpoint.pt"
    train._save_checkpoint(path, payload)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    set_rng_states = Mock()
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_rng_states)
    assert train._load_checkpoint(path, model, encoder, optimizer, torch.device("cpu")) == 40000
    set_rng_states.assert_called_once()
    actual_states = set_rng_states.call_args.args[0]
    assert len(actual_states) == min(saved_device_count, 2)
    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_states, saved_states[:2]))


class _TinyConditionEncoder(nn.Linear):
    def __init__(self, **kwargs):
        super().__init__(2, 2)

    def forward(self, hamer_tokens, *args):
        return super().forward(hamer_tokens)


@pytest.fixture
def main_harness(tmp_path, monkeypatch):
    model_factory = Mock(side_effect=lambda **kwargs: nn.Linear(2, 2))
    monkeypatch.setattr(train, "MMDiT", model_factory)
    monkeypatch.setattr(train, "ConditionEncoder", _TinyConditionEncoder)
    monkeypatch.setattr(train, "ManoWrapper", nn.Identity)
    monkeypatch.setattr(train, "VGGTOmegaCameraTokenWrapper", lambda **kwargs: nn.Identity())
    for name in ("FeatureWorker", "DetectorWorker", "Hot3DRectifier", "HaMeRTokenCache", "HandBoxCache"):
        monkeypatch.setattr(train, name, Mock())
    monkeypatch.setattr(train.atexit, "register", Mock())
    monkeypatch.setattr(train, "_set_seed", Mock())
    monkeypatch.setattr(torch, "set_num_threads", Mock())
    monkeypatch.setattr(torch, "set_num_interop_threads", Mock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "_split_clip_ids", Mock(return_value=["000001"]))
    sample = SimpleNamespace(
        target_mano=ManoFlowState(
            translation=torch.zeros(1, 2, 1, 3),
            root_rotation=torch.zeros(1, 2, 1, 6),
            local_rotation=torch.zeros(1, 2, 1, 15, 6),
        ),
        target_camera=CameraFlowState(
            translation=torch.zeros(1, 1, 3), rotation=torch.zeros(1, 1, 6)
        ),
        hamer_tokens=torch.ones(1, 2),
        vggt_tokens=torch.ones(1, 2),
        keypoints=torch.zeros(1, 2, 1, 21, 3),
        bbox=torch.zeros(1, 2, 1, 4),
        hamer_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        bbox_valid=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    adapter = Mock()
    adapter.sample.return_value = sample
    monkeypatch.setattr(train, "make_adapter", Mock(return_value=adapter))
    flow_time = Mock(side_effect=lambda batch, sigma, **kwargs: torch.full((batch,), 0.5))
    monkeypatch.setattr(train, "sample_flow_time", flow_time)
    step_mock = Mock(
        side_effect=lambda **kwargs: (kwargs["model"](kwargs["conditions"]).square().mean(), {})
    )
    monkeypatch.setattr(train, "train_step", step_mock)
    load_checkpoint = Mock(return_value=40000)
    monkeypatch.setattr(train, "_load_checkpoint", load_checkpoint)
    monkeypatch.setattr(train, "_save_checkpoint", Mock())
    launch_evaluation = Mock(side_effect=AssertionError("Evaluation must not be launched"))
    monkeypatch.setattr(train, "_launch_evaluation", launch_evaluation)
    selected_range = Mock(return_value=[])
    monkeypatch.setattr(train, "range", selected_range, raising=False)
    argv = [
        "train", "--max-steps", "80000",
        "--sigma-start", "1", "--sigma-end", "0.5", "--no-eval", "--device", "cpu",
    ]
    monkeypatch.setattr(train.sys, "argv", argv)
    return SimpleNamespace(
        argv=argv, selected_range=selected_range, flow_time=flow_time,
        train_step=step_mock, load_checkpoint=load_checkpoint,
        model_factory=model_factory, launch_evaluation=launch_evaluation,
    )


def test_main_resumed_sigma_decay_keeps_auxiliary_losses_enabled(main_harness, tmp_path) -> None:
    harness = main_harness
    resume = tmp_path / "mock_checkpoint.pt"
    harness.argv.extend(["--resume", str(resume), "--sigma-decay-start-step", "40000"])
    harness.selected_range.return_value = [40000, 60000, 79999]

    train.main()

    harness.load_checkpoint.assert_called_once()
    assert harness.load_checkpoint.call_args.args[0] == resume
    harness.selected_range.assert_called_once_with(40000, 80000)
    sigmas = [call.args[1] for call in harness.flow_time.call_args_list]
    assert sigmas == pytest.approx([1.0, 0.7499937498437461, 0.5], abs=1e-12)
    assert [call.kwargs["auxiliary_enabled"] for call in harness.train_step.call_args_list] == [True] * 3
    harness.launch_evaluation.assert_not_called()


def test_main_default_schedule_retains_original_denominator_and_uniform_phase(main_harness) -> None:
    harness = main_harness
    harness.selected_range.return_value = [9999, 10000, 40000, 60000, 79999]

    train.main()

    harness.load_checkpoint.assert_not_called()
    harness.selected_range.assert_called_once_with(0, 80000)
    sigmas = [call.args[1] for call in harness.flow_time.call_args_list]
    assert sigmas == pytest.approx([1.0, 11 / 14, 9 / 14, 0.5000071428571429], abs=1e-12)
    calls = harness.train_step.call_args_list
    assert [call.kwargs["auxiliary_enabled"] for call in calls] == [False, True, True, True, True]
    assert 0 <= calls[0].kwargs["flow_time"].min().item() <= 1
    assert calls[0].kwargs["flow_time"].shape == (1,)


@pytest.mark.parametrize(
    ("anchor", "steps", "expected_sigmas"),
    [
        (40000, [9999, 10000, 39999, 40000, 79999], [1.0, 1.0, 1.0, 0.5]),
        (10000, [10000, 79999], [1.0, 0.5]),
        (79999, [79998, 79999], [1.0, 1.0]),
    ],
)
def test_main_explicit_anchor_clamps_progress_and_handles_short_decay(
    main_harness, anchor, steps, expected_sigmas
) -> None:
    harness = main_harness
    harness.argv.extend(["--sigma-decay-start-step", str(anchor)])
    harness.selected_range.return_value = steps

    train.main()

    sigmas = [call.args[1] for call in harness.flow_time.call_args_list]
    assert sigmas == pytest.approx(expected_sigmas, abs=1e-12)
    assert harness.train_step.call_count == len(steps)


@pytest.mark.parametrize(
    ("anchor", "uniform_steps"),
    [(-1, 0), (9999, 10000), (80000, 10000), (80001, 10000), ("40000.5", 10000)],
)
def test_main_rejects_invalid_sigma_anchor_before_building_models(
    main_harness, capsys, anchor, uniform_steps
) -> None:
    harness = main_harness
    harness.argv.extend([
        "--sigma-decay-start-step", str(anchor), "--uniform-steps", str(uniform_steps)
    ])

    with pytest.raises(SystemExit) as error:
        train.main()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--sigma-decay-start-step" in stderr
    assert "unrecognized arguments" not in stderr
    harness.model_factory.assert_not_called()
    harness.load_checkpoint.assert_not_called()
    harness.selected_range.assert_not_called()
