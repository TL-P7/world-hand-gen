"""Persistent ViTDet/ViTPose hand-box worker."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import os

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
        raise EOFError("Detector worker exited; see its stderr for the cause")
    payload = stream.read(struct.unpack("!Q", header)[0])
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


class DetectorWorker(nn.Module):
    """Synchronous parent-side interface for the external detector process."""

    def __init__(
        self,
        python,
        repository,
        detector_checkpoint,
        device="cuda",
        image_size=384,
        precision="fp16",
    ):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        child_env = os.environ.copy()
        for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            child_env.pop(variable, None)
        child_device = str(device)
        if child_device.startswith("cuda:"):
            child_env["CUDA_VISIBLE_DEVICES"] = child_device.split(":", 1)[1]
            child_device = "cuda:0"
        command = [
            str(python), "-m", "src.train.detector_worker",
            "--repository", str(Path(repository).resolve()),
            "--detector-checkpoint", str(Path(detector_checkpoint).resolve()),
            "--device", child_device,
            "--image-size", str(image_size),
            "--precision", precision,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            self.metadata = _receive(self.process.stdout)
        except BaseException:
            self.close()
            raise

    def forward(self, images: torch.Tensor):
        _send(self.process.stdin, images.detach().cpu())
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
    parser.add_argument("--repository", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from src.model.wrappers.hamer_detector_wrapper import HamerHandBoxDetector
    with contextlib.redirect_stdout(sys.stderr):
        model = HamerHandBoxDetector(
            repository=args.repository,
            detector_checkpoint=args.detector_checkpoint,
            device=args.device,
            image_size=args.image_size,
            precision=args.precision,
        )
    _send(sys.stdout.buffer, {"mode": f"vitdet_h_{args.precision}_{args.image_size}_no_square_pad"})
    while True:
        try:
            images = _receive(sys.stdin.buffer)
        except EOFError:
            break
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            predictions = model(images)
        _send(sys.stdout.buffer, predictions)


if __name__ == "__main__":
    main()
