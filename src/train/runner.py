"""Dataset-neutral training runners for selectable objectives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.model.flow.joint import JointFlowModel, JointFlowState
from src.train.joint_flow import train_joint_step


def parse_key_values(values: list[str]) -> dict[str, Any]:
    """Parse opaque adapter configuration without interpreting dataset splits."""
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


def train_joint(
    *,
    adapter,
    steps: int,
    batch_size: int,
    window: int,
    device: str,
    seed: int,
    log_every: int,
    save_every: int,
    output_dir: Path,
    model_dim: int = 128,
    depth: int = 2,
    heads: int = 4,
    learning_rate: float = 5e-4,
) -> list[dict[str, float]]:
    """Run a direct public-joint flow objective for any adapter."""
    if steps < 1 or log_every < 1 or save_every < 1:
        raise ValueError("steps, log_every, and save_every must be positive")
    torch.manual_seed(seed)
    target_sample = adapter.sample(seed=seed, batch_size=batch_size, window=window)
    if target_sample.target_joints3d is None or target_sample.joint_valid is None:
        raise ValueError("selected joint objective requires adapter joint supervision")
    target = JointFlowState(target_sample.target_joints3d.to(device))
    valid = target_sample.joint_valid.to(device)
    model = JointFlowModel(model_dim=model_dim, depth=depth, heads=heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    config = {
        "objective": "joints3d",
        "steps": steps,
        "batch_size": batch_size,
        "window": window,
        "seed": seed,
        "adapter": adapter.describe(),
        "sample": target_sample.metadata,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    log_path = output_dir / "metrics.jsonl"
    history: list[dict[str, float]] = []
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        time = torch.rand(batch_size, device=device, generator=generator)
        noise = JointFlowState(torch.randn(target.joints.shape, device=device, generator=generator))
        optimizer.zero_grad(set_to_none=True)
        value, _ = train_joint_step(model, target, valid, time, noise)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == 1 or step == steps:
            record = {"step": float(step), "joints3d": float(value.detach().cpu())}
            history.append(record)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"step={step} joints3d={record['joints3d']:.6f}", flush=True)
        if step % save_every == 0 or step == steps:
            torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), **config},
                       output_dir / "checkpoints" / f"step_{step:08d}.pt")
    return history
