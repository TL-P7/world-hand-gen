"""Fixed-configuration training loop for the world-hand-gen prototype.

All runtime choices are command-line arguments rather than YAML. The loop uses
T=64 windows, a global batch size of 16 (four local windows on each of four
DDP ranks), fixed fp32, AdamW with constant 5e-5 learning rate, fixed seed,
10-step logging, and resumable checkpoints.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from builtins import range as builtin_range

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.config import (
    CACHE_ROOT,
    DATA_ROOT,
    DETECTOR_CHECKPOINT,
    DETECTOR_PYTHON,
    EXPERIMENT_ROOT,
    HAMER_CHECKPOINT,
    HAMER_PYTHON,
    HAMER_REPOSITORY,
    VGGT_CHECKPOINT,
    VGGT_REPOSITORY,
    PROJECT_ROOT,
)
from src.data.adapters import Hot3DAdapter, make_adapter
from src.data.hot3d import Hot3DRectifier
from src.data.hot3d.cache import HaMeRTokenCache, VGGTOmegaTokenCache
from src.data.hot3d.handbox import HandBoxCache
from src.data.hot3d.splits import partition_clips
from src.data.contract import WindowSample
from src.model.flow.condition import ConditionEncoder
from src.model.flow.mmdit import MMDiT
from src.model.flow.state import CameraFlowState, ManoFlowState
from src.model.wrappers import ManoWrapper
from src.model.wrappers.vggt_omega_wrapper import VGGTOmegaCameraTokenWrapper
from src.train.feature_worker import FeatureWorker
from src.train.detector_worker import DetectorWorker
from src.train.flow_sampling import sample_flow_time, sample_noise
from src.train.sample import collate_samples, sample_batch, sample_window
from src.train.step import DEFAULT_WEIGHTS, train_step
from src.train.runner import parse_key_values, parse_losses, train_joint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distributed_info(device_arg: str, expected_world_size: int | None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(f"expected world size {expected_world_size}, got {world_size}")
    if distributed:
        backend = "gloo" if device_arg == "cpu" else "nccl"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend)
    if device_arg == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else device_arg)
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device


def _unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _is_main(rank: int) -> bool:
    return rank == 0


def _reduce_scalar(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


class _OfflineWorker:
    """Placeholder used when all detector/HaMeR/VGGT features are cached."""

    model_cfg = None
    offline = True

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("offline feature cache miss; run the precompute stage first")

    def close(self):
        return None


def _split_clip_ids(split: str) -> list[str]:
    """Return the deterministic local split of the mounted train_aria clips."""
    official = json.loads((DATA_ROOT / "clip_splits.json").read_text())
    mounted = [f"{clip_id:06d}" for clip_id in official["train"]["Aria"]]
    train, test = partition_clips(mounted, train_count=1272, seed=2026)
    return train if split == "train" else test


def _move_mano(state: ManoFlowState, device: torch.device) -> ManoFlowState:
    return state.map(lambda value: value.to(device))


def _move_camera(state: CameraFlowState, device: torch.device) -> CameraFlowState:
    return state.map(lambda value: value.to(device))


def _checkpoint_payload(step, model, condition_encoder, optimizer, seed, **metadata):
    return {
        "step": step,
        "model": _unwrap(model).state_dict(),
        "condition_encoder": _unwrap(condition_encoder).state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": seed,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **metadata,
    }


def _save_checkpoint(path, payload):
    """Atomically publish a checkpoint so an evaluator never reads a partial file."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _launch_evaluation(args, checkpoint: Path, step: int) -> None:
    output_dir = args.experiment_dir / "eval" / f"step_{step:08d}"
    log_path = args.experiment_dir / "eval" / f"step_{step:08d}.log"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.experiment_dir / "eval" / ".evaluation.lock"
    command = [
        "flock", "--exclusive", str(lock_path),
        sys.executable, "-m", "src.eval.run",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(output_dir),
        "--device", str(args.eval_device),
        "--hamer-device", str(args.eval_hamer_device),
        "--detector-device", str(args.eval_detector_device),
        "--vggt-device", str(args.eval_vggt_device),
    ]
    with log_path.open("ab") as log:
        subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)


def _load_checkpoint(path, model, condition_encoder, optimizer, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _unwrap(model).load_state_dict(payload["model"])
    _unwrap(condition_encoder).load_state_dict(payload["condition_encoder"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"][:torch.cuda.device_count()])
    return int(payload["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="hot3d")
    parser.add_argument("--dataset-config", action="append", default=[])
    parser.add_argument("--losses", default=None)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--expected-world-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--auxiliary-start-step", type=int, default=10000)
    parser.add_argument("--uniform-steps", type=int, default=10000)
    parser.add_argument("--sigma-start", type=float, default=1.0)
    parser.add_argument("--sigma-end", type=float, default=0.5)
    parser.add_argument(
        "--sigma-decay-start-step", type=int, default=None,
        help="Completed-step index where sigma starts decaying; reaches sigma-end on the final update",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--no-eval", dest="eval_after_checkpoint", action="store_false")
    parser.add_argument("--eval-after-checkpoint", dest="eval_after_checkpoint", action="store_true")
    parser.set_defaults(eval_after_checkpoint=True)
    parser.add_argument("--profile", action="store_true", help="Log synchronized sample/train seconds per step")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--hamer-device", default=None)
    parser.add_argument("--vggt-device", default=None)
    parser.add_argument("--vggt-image-resolution", type=int, default=512)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--condition-device-offset", type=int, default=None)
    parser.add_argument("--offline-features", action="store_true")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--window-start", type=int, default=None)
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--eval-hamer-device", default="cuda:1")
    parser.add_argument("--eval-detector-device", default="cuda:2")
    parser.add_argument("--eval-vggt-device", default="cuda:3")
    args = parser.parse_args()
    if args.dataset != "hot3d":
        losses = parse_losses(args.losses or "joints3d=1.0")
        if set(name for name, weight in losses.items() if weight > 0) != {"joints3d"}:
            parser.error("non-HOT3D adapters currently require exactly joints3d as the positive loss")
        adapter = make_adapter(args.dataset, **parse_key_values(args.dataset_config))
        train_joint(
            adapter=adapter,
            steps=args.max_steps,
            batch_size=args.global_batch_size,
            window=args.window,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            save_every=args.save_every,
            output_dir=args.experiment_dir,
        )
        return
    args.cache_root = CACHE_ROOT
    if args.window < 1:
        parser.error("--window must be positive")

    if args.sigma_decay_start_step is not None and not (
        max(args.uniform_steps, 0) <= args.sigma_decay_start_step < args.max_steps
    ):
        parser.error("--sigma-decay-start-step must be nonnegative, >= uniform-steps and < max-steps")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    distributed, rank, local_rank, world_size, device = _distributed_info(
        args.device, args.expected_world_size
    )
    local_batch_size = args.global_batch_size // world_size
    if args.global_batch_size < 1 or args.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    _set_seed(args.seed + rank)
    if _is_main(rank):
        print(
            f"distributed world_size={world_size} global_batch_size={args.global_batch_size} "
            f"local_batch_size={local_batch_size}",
            flush=True,
        )
    else:
        print(f"rank={rank} local_rank={local_rank} device={device}", flush=True)
    if _is_main(rank):
        args.experiment_dir.mkdir(parents=True, exist_ok=True)
        (args.experiment_dir / "checkpoints").mkdir(exist_ok=True)

    model = MMDiT(model_dim=512, depth=8, heads=8).to(device)
    condition_encoder = ConditionEncoder(model_dim=512).to(device)
    if distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
        condition_encoder = DistributedDataParallel(condition_encoder, **ddp_kwargs)
    mano = ManoWrapper().to(device).eval()
    condition_device = device
    if args.condition_device_offset is not None and device.type == "cuda":
        condition_device = torch.device(
            f"cuda:{args.condition_device_offset + local_rank}"
        )
    if args.offline_features:
        hamer = _OfflineWorker()
        detector = _OfflineWorker()
        vggt = VGGTOmegaTokenCache(cache_root=args.cache_root, vggt=None)
    else:
        hamer_device = args.hamer_device or str(condition_device)
        if distributed and hamer_device == "cuda":
            hamer_device = str(condition_device)
        hamer = FeatureWorker(
            HAMER_PYTHON, "hamer", HAMER_REPOSITORY, HAMER_CHECKPOINT, device=hamer_device
        )
        atexit.register(hamer.close)
        detector_device = args.detector_device or str(condition_device)
        if distributed and detector_device == "cuda":
            detector_device = str(condition_device)
        detector = DetectorWorker(
            DETECTOR_PYTHON, HAMER_REPOSITORY, DETECTOR_CHECKPOINT,
            device=detector_device, image_size=384, precision="fp16",
        )
        atexit.register(detector.close)
        vggt = VGGTOmegaCameraTokenWrapper(
            repository=VGGT_REPOSITORY, checkpoint=VGGT_CHECKPOINT,
            image_resolution=args.vggt_image_resolution,
        ).to(condition_device).eval()
    rectifier = Hot3DRectifier(output_size=384)
    cache = HaMeRTokenCache(cache_root=args.cache_root, hamer=hamer)
    handbox_cache = HandBoxCache(
        cache_root=args.cache_root, detector=detector, batch_size=args.detector_batch_size,
    )

    selected_weights = parse_losses(args.losses)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(condition_encoder.parameters()),
        lr=5e-5,
        weight_decay=1e-2,
        betas=(0.9, 0.99),
    )
    start_step = 0
    if args.resume is not None:
        start_step = _load_checkpoint(args.resume, model, condition_encoder, optimizer, device)
        if _is_main(rank):
            print(f"resumed checkpoint={args.resume} step={start_step} max_steps={args.max_steps}", flush=True)

    adapter_config = parse_key_values(args.dataset_config)
    adapter_config.setdefault("split", args.split)
    if args.clip_id is not None:
        adapter_config.setdefault("clip_id", args.clip_id)
    if args.window_start is not None:
        adapter_config.setdefault("window_start", args.window_start)
    adapter_config.update(
        clip_ids=None,
        hamer_cache=cache,
        handbox_cache=handbox_cache,
        vggt=vggt,
        rectifier=rectifier,
        window=args.window,
    )
    adapter = make_adapter("hot3d", **adapter_config)
    running: dict[str, float] = {}
    try:
        for step in range(start_step, args.max_steps):
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = perf_counter()
            seeds = [
                args.seed + (step * world_size + rank) * local_batch_size + index
                for index in builtin_range(local_batch_size)
            ]
            sample = adapter.sample(
                seed=seeds[0], batch_size=local_batch_size, window=args.window
            )
            if args.profile and device.type == "cuda":
                torch.cuda.synchronize(device)
            sample_end = perf_counter()
            target_mano = _move_mano(sample.target_mano, device)
            target_camera = _move_camera(sample.target_camera, device)
            hamer_tokens = sample.hamer_tokens.to(device)
            vggt_tokens = sample.vggt_tokens.to(device)
            keypoints = sample.keypoints.to(device)
            bbox = sample.bbox.to(device)
            batch = hamer_tokens.shape[0]
            conditions = condition_encoder(
                hamer_tokens,
                vggt_tokens,
                bbox,
                sample.hamer_valid.to(device),
                torch.ones(batch, args.window, dtype=torch.bool, device=device),
                sample.bbox_valid.to(device),
                keypoints,
            )

            generator = torch.Generator(device=device).manual_seed(
                args.seed + 1000003 * (step * world_size + rank)
            )
            if step < args.uniform_steps:
                flow_time = torch.rand(batch, device=device, generator=generator)
            else:
                if args.sigma_decay_start_step is None:
                    progress = min((step - args.uniform_steps) / max(args.max_steps - args.uniform_steps, 1), 1.0)
                else:
                    progress = min(max(
                        (step - args.sigma_decay_start_step)
                        / max(args.max_steps - args.sigma_decay_start_step - 1, 1), 0.0,
                    ), 1.0)
                sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
                if _is_main(rank) and (step == start_step or (step + 1) % args.log_every == 0):
                    print(f"schedule step={step + 1} sigma={sigma:.8f}", flush=True)
                flow_time = sample_flow_time(batch, sigma, generator=generator).to(device)
            noise_mano, noise_camera = sample_noise(target_mano, target_camera, generator=generator)

            optimizer.zero_grad(set_to_none=True)
            loss, terms = train_step(
                model=model,
                mano_wrapper=mano,
                target_mano=target_mano,
                target_camera=target_camera,
                conditions=conditions,
                flow_time=flow_time,
                noise_mano=noise_mano,
                noise_camera=noise_camera,
                        weights=selected_weights,

                auxiliary_enabled=step >= args.auxiliary_start_step,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(condition_encoder.parameters()), 1.0
            )
            optimizer.step()

            if args.profile:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = perf_counter() - step_start
                if _is_main(rank):
                    print(
                        f"profile step={step + 1} sample={sample_end - step_start:.3f}s "
                        f"train={elapsed - (sample_end - step_start):.3f}s total={elapsed:.3f}s",
                        flush=True,
                    )

            running["total"] = running.get("total", 0.0) + _reduce_scalar(loss.item(), device, world_size)
            for name, value in terms.items():
                running[name] = running.get(name, 0.0) + _reduce_scalar(value.item(), device, world_size)
            if (step + 1) % args.log_every == 0 and _is_main(rank):
                count = args.log_every
                values = " ".join(f"{name}={value / count:.5f}" for name, value in running.items())
                print(f"step={step + 1} {values}", flush=True)
                running = {}
            if (step + 1) % args.save_every == 0 and _is_main(rank):
                path = args.experiment_dir / "checkpoints" / f"step_{step + 1:08d}.pt"
                _save_checkpoint(
                    path,
                    _checkpoint_payload(
                        step + 1,
                        model,
                        condition_encoder,
                        optimizer,
                        args.seed,
                        world_size=world_size,
                        global_batch_size=args.global_batch_size,
                        local_batch_size=local_batch_size,
                    ),
                )
                if args.eval_after_checkpoint:
                    _launch_evaluation(args, path, step + 1)
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    main()
