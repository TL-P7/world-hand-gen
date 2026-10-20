"""Command-line entry point for dataset-neutral objective validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.adapters import make_adapter
from src.train.runner import parse_key_values, parse_losses, train_mano


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", action="append", default=[])
    parser.add_argument("--losses", default="joints3d=1.0")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=10000)
    parser.add_argument("--batch-size", "--global-batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    args = parser.parse_args()
    losses = parse_losses(args.losses)
    adapter = make_adapter(args.dataset, **parse_key_values(args.dataset_config))
    train_mano(
        adapter=adapter, steps=args.steps, batch_size=args.batch_size,
        window=args.window, device=args.device, seed=args.seed,
        log_every=args.log_every, save_every=args.save_every,
        output_dir=args.output_dir, weights=losses,
        model_dim=512, depth=8, heads=8,
    )


if __name__ == "__main__":
    main()
