"""Dataset-neutral MANO-flow training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.model.flow.condition import ConditionEncoder
from src.model.flow.mmdit import MMDiT
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.wrappers.mano_wrapper import ManoWrapper
from src.train.flow_sampling import sample_flow_time, sample_noise_shapes
from src.train.step import train_step


def parse_key_values(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"dataset config must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("dataset config keys must not be empty")
        if value.lower() in {"none", "null"}:
            result[key] = None
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def parse_losses(specification: str | None) -> dict[str, float]:
    if not specification:
        return {"flow": 1.0, "camera_flow": 1.0, "joints3d": 0.5,
                "acceleration": 0.02, "camera_pose": 0.1, "camera_trajectory": 0.1}
    result: dict[str, float] = {}
    for item in specification.split(","):
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"loss must be name=weight, got {item!r}")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"loss weight must be nonnegative: {item!r}")
        result[name] = weight
    return result


def train_mano(
    *, adapter, steps: int, batch_size: int, window: int, device: str, seed: int,
    log_every: int, save_every: int, output_dir: Path, weights: dict[str, float],
    model_dim: int = 512, depth: int = 8, heads: int = 8,
) -> list[dict[str, float]]:
    """Train the production MMDiT MANO-flow output with configured losses."""
    if steps < 1 or log_every < 1 or save_every < 1:
        raise ValueError("steps, log_every, and save_every must be positive")
    torch.manual_seed(seed)
    sample = adapter.sample(seed=seed, batch_size=batch_size, window=window)
    model = MMDiT(model_dim=model_dim, depth=depth, heads=heads).to(device)
    encoder = ConditionEncoder(model_dim=model_dim).to(device)
    mano = ManoWrapper().to(device).eval()
    optimizer = torch.optim.AdamW([*model.parameters(), *encoder.parameters()], lr=5e-5, weight_decay=1e-2)
    target_mano = sample.target_mano.map(lambda value: value.to(device)) if sample.target_mano is not None else None
    target_camera = sample.target_camera.map(lambda value: value.to(device)) if sample.target_camera is not None else None
    target_joints = sample.target_joints3d.to(device) if sample.target_joints3d is not None else None
    valid = sample.joint_valid.to(device) if sample.joint_valid is not None else None
    hamer = sample.hamer_tokens.to(device)
    vggt = sample.vggt_tokens.to(device)
    bbox = sample.bbox.to(device)
    conditions = encoder(hamer, vggt, bbox, sample.hamer_valid.to(device), sample.vggt_valid.to(device), sample.bbox_valid.to(device), sample.keypoints.to(device))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    config = {"model_output": "ManoFlowState", "losses": weights, "steps": steps, "batch_size": batch_size, "window": window, "seed": seed, "adapter": adapter.describe(), "sample": sample.metadata}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    history = []
    for step in range(1, steps + 1):
        flow_time = sample_flow_time(batch_size, 1.0, generator=generator).to(device)
        noise_mano, noise_camera = sample_noise_shapes(
            (batch_size, 2, window, 3), (batch_size, 2, window, 6), (batch_size, 2, window, 15, 6),
            (batch_size, window, 6), (batch_size, window, 3), device=device, generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        value, terms = train_step(
            model=model, mano_wrapper=mano, target_mano=target_mano, target_camera=target_camera,
            target_joints3d=target_joints, joint_valid=valid, conditions=conditions,
            flow_time=flow_time, noise_mano=noise_mano, noise_camera=noise_camera,
            weights=weights, auxiliary_enabled=True,
        )
        value.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *encoder.parameters()], 1.0)
        optimizer.step()
        if step % log_every == 0 or step == 1 or step == steps:
            record = {"step": float(step), "total": float(value.detach().cpu())}
            record.update({name: float(term.detach().cpu()) for name, term in terms.items()})
            history.append(record)
            with (output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("step=" + str(step) + " " + " ".join(f"{k}={v:.6f}" for k, v in record.items() if k != "step"), flush=True)
        if step % save_every == 0 or step == steps:
            torch.save({"step": step, "model": model.state_dict(), "condition_encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(), **config}, output_dir / "checkpoints" / f"step_{step:08d}.pt")
    return history
