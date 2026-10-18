"""Shared first-frame alignment and scale fitting for metrics.

Prediction and target are each aligned to their own first frame so the two
sequences start from a consistent origin. ``fit_scale`` returns the positive
scalar that best scales the prediction motion onto the target motion, used to
measure whether the model learns the correct motion magnitude.
"""

from __future__ import annotations

import torch
from torch import Tensor


def align_translation(x: Tensor, t0: Tensor) -> Tensor:
    """Subtract the first-frame translation so sequences start at the origin."""
    return x - t0


def align_rigid(x: Tensor, t0: Tensor, r0: Tensor) -> Tensor:
    """Subtract first-frame translation and undo first-frame rotation."""
    return torch.einsum("...ij,...j->...i", r0.transpose(-2, -1), x - t0)


def fit_scale(prediction: Tensor, target: Tensor) -> Tensor:
    """Least-squares positive scalar mapping prediction onto target motion."""
    numerator = (prediction * target).sum()
    denominator = (prediction * prediction).sum()
    return numerator / denominator.clamp_min(1e-12)
