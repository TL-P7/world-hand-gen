"""Serialized, resumable full test244 evaluations for an imported and continued run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.config import PROJECT_ROOT
from src.eval.protocol import PROTOCOL, evaluate_saved


STEPS = (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)


def _completed(output_dir: Path, checkpoint: Path, step: int, *, protocol=True) -> bool:
    try:
        request = json.loads((output_dir / "request.json").read_text())
        summary = json.loads((output_dir / "summary.json").read_text())
        predictions = output_dir / "predictions.pt"
        if not predictions.is_file() or predictions.stat().st_size == 0:
            return False
        with (output_dir / "records.csv").open(newline="") as handle:
            record_count = sum(1 for _ in csv.DictReader(handle))
        return (
            Path(request["checkpoint"]).resolve() == checkpoint.resolve()
            and request["checkpoint_step"] == summary["checkpoint_step"] == step
            and summary["split"] == "test244"
            and summary["clips"] == 244
            and summary["windows"] == record_count == 488
            and summary["smoke_only"] is False
            and len(summary["metrics"]) == 25
            and (not protocol or (
                summary.get("protocol") == PROTOCOL
                and summary.get("filter_offscreen") is True
                and summary.get("hand_windows", 0) + summary.get("excluded_windows", 0) == 488
            ))
        )
    except (FileNotFoundError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _output_dir(eval_dir: Path, step: int) -> Path:
    canonical = eval_dir / f"step_{step:08d}"
    if not canonical.exists():
        return canonical
    retry = 1
    while (eval_dir / f"step_{step:08d}_retry_{retry:02d}").exists():
        retry += 1
    return eval_dir / f"step_{step:08d}_retry_{retry:02d}"


def _free_memory(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU for the evaluator and workers")
    parser.add_argument("--minimum-free-mib", type=int, default=26000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.gpu < 0 or args.minimum_free_mib < 1 or args.poll_seconds < 1 or args.max_attempts < 1:
        parser.error("GPU index must be nonnegative; memory threshold, polling interval, and attempts must be positive")

    eval_dir = args.experiment_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / ".queue.lock").open("w") as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("an evaluation queue is already running in this experiment")
        failures = []
        for step in STEPS:
            source = args.source_dir if step <= 12000 else args.experiment_dir / "checkpoints"
            checkpoint = source / f"step_{step:08d}.pt"
            phase = "pre-change" if step < 12000 else "resume-12000"
            while not checkpoint.is_file():
                print(f"step={step} phase={phase} waiting for checkpoint={checkpoint}", flush=True)
                time.sleep(args.poll_seconds)
            attempts = 0
            automatic_waits = 0
            while True:
                with (eval_dir / ".evaluation.lock").open("a") as shared_lock:
                    fcntl.flock(shared_lock, fcntl.LOCK_EX)
                    candidates = [eval_dir / f"step_{step:08d}"]
                    candidates += sorted(eval_dir.glob(f"step_{step:08d}_retry_*"))
                    complete = [output for output in candidates if _completed(output, checkpoint, step)]
                    if complete:
                        print(f"step={step} phase={phase} completed checkpoint={checkpoint}", flush=True)
                        break
                    reusable = [output for output in candidates if _completed(output, checkpoint, step, protocol=False)]
                    if reusable:
                        output = reusable[0]
                        protocol_dir = output / "protocol"
                        if not protocol_dir.exists():
                            evaluate_saved(output / "predictions.pt", protocol_dir, expected_windows=488)
                        if (protocol_dir / "summary.json").is_file():
                            summary = json.loads((protocol_dir / "summary.json").read_text())
                            if summary.get("protocol") != PROTOCOL or summary.get("checkpoint_step") != step:
                                raise ValueError(f"invalid protocol evaluation: {protocol_dir}")
                            print(f"step={step} phase={phase} recomputed protocol checkpoint={checkpoint}", flush=True)
                            break
                        raise ValueError(f"incomplete protocol evaluation: {protocol_dir}")
                    canonical = candidates[0]
                    if step > 12000 and not canonical.exists() and automatic_waits < 2:
                        automatic_waits += 1
                        print(f"step={step} phase={phase} waiting for automatic evaluation", flush=True)
                    elif _free_memory(args.gpu) < args.minimum_free_mib:
                        print(f"step={step} phase={phase} waiting for GPU {args.gpu} memory", flush=True)
                    else:
                        output_dir = _output_dir(eval_dir, step)
                        log_path = eval_dir / f"{output_dir.name}.log"
                        env = os.environ.copy()
                        for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                            env.pop(variable, None)
                        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                        print(f"step={step} phase={phase} gpu={args.gpu} checkpoint={checkpoint} output={output_dir}", flush=True)
                        command = [
                            sys.executable, "-m", "src.eval.run",
                            "--checkpoint", str(checkpoint), "--output-dir", str(output_dir),
                            "--device", "cuda:0", "--hamer-device", f"cuda:{args.gpu}",
                            "--detector-device", f"cuda:{args.gpu}",
                            "--vggt-device", f"cuda:{args.gpu}",
                        ]
                        with log_path.open("x") as log:
                            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env,
                                                    stdout=log, stderr=subprocess.STDOUT)
                        attempts += 1
                        print(f"step={step} phase={phase} exit={result.returncode} output={output_dir}", flush=True)
                        if _completed(output_dir, checkpoint, step):
                            break
                        if attempts >= args.max_attempts:
                            failures.append(step)
                            print(f"step={step} phase={phase} FAILED after {attempts} attempts", flush=True)
                            break
                time.sleep(args.poll_seconds)
        if failures:
            print(f"failed checkpoint evaluations: {failures}", flush=True)
            sys.exit(1)
        print("all requested checkpoint evaluations completed", flush=True)


if __name__ == "__main__":
    main()
