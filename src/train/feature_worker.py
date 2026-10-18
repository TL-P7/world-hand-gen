"""Persistent token inference in each external model's Python environment."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys

import torch
from torch import nn

from src.config import PROJECT_ROOT


def _send(stream, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _receive(stream):
    header = stream.read(8)
    if len(header) != 8:
        raise EOFError("Token worker exited; see its stderr for the cause")
    payload = stream.read(struct.unpack("!Q", header)[0])
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


class FeatureWorker(nn.Module):
    """Synchronous tensor interface to one long-lived inference process.

    Inputs/outputs cross the process boundary on CPU. Separate workers may be
    called concurrently; a single worker serves one request at a time.
    """

    def __init__(self, python, kind, repository, checkpoint, device="cuda"):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        child_env = os.environ.copy()
        for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            child_env.pop(variable, None)
        child_device = str(device)
        if child_device.startswith("cuda:"):
            child_env["CUDA_VISIBLE_DEVICES"] = child_device.split(":", 1)[1]
            child_device = "cuda:0"
        self.process = subprocess.Popen(
            [str(python), "-m", "src.train.feature_worker", "--kind", kind,
             "--repository", str(Path(repository).resolve()),
             "--checkpoint", str(Path(checkpoint).resolve()), "--device", child_device],
            cwd=PROJECT_ROOT, env=child_env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        try:
            metadata = _receive(self.process.stdout)
        except BaseException:
            self.close()
            raise
        if kind == "hamer":
            from yacs.config import CfgNode
            self.model_cfg = CfgNode({"MODEL": metadata["model_config"]})

    def forward(self, tensor):
        _send(self.process.stdin, tensor.detach().cpu())
        return _receive(self.process.stdout)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait()
        self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("hamer", "vggt"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        if args.kind == "hamer":
            from src.model.wrappers.hamer_wrapper import HaMeRDecoderTokenWrapper
            model = HaMeRDecoderTokenWrapper(repository=args.repository, checkpoint=args.checkpoint)
            cfg = model.model_cfg.MODEL
            metadata = {"model_config": {
                "IMAGE_SIZE": cfg.IMAGE_SIZE, "IMAGE_MEAN": list(cfg.IMAGE_MEAN),
                "IMAGE_STD": list(cfg.IMAGE_STD),
                "BBOX_SHAPE": list(cfg.BBOX_SHAPE),
            }}
        else:
            from src.model.wrappers.vggt_omega_wrapper import VGGTOmegaCameraTokenWrapper
            model = VGGTOmegaCameraTokenWrapper(repository=args.repository, checkpoint=args.checkpoint)
            metadata = {}
        model.to(args.device).eval()
    _send(output, metadata)
    while True:
        try:
            tensor = _receive(sys.stdin.buffer)
        except EOFError:
            break
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            tokens = model(tensor.to(args.device)).float().cpu()
        _send(output, tokens)


if __name__ == "__main__":
    main()
