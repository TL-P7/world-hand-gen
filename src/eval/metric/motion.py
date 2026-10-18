"""Temporal hand metrics: velocity, acceleration, and endpoint drift.

Unit frame spacing is assumed (``dt = 1``). Positions are converted to
millimetres before differencing, so velocity is mm/s and acceleration mm/s^2.
"""

from __future__ import annotations

import torch
from torch import Tensor

_MM = 1000.0


def _velocity(values: Tensor) -> Tensor:
    """First forward difference ``x[t+1] - x[t]`` on the time axis."""
    return values[:, :, 1:] - values[:, :, :-1]


def _acceleration(values: Tensor) -> Tensor:
    """Second forward difference ``x[t+2] - 2x[t+1] + x[t]``."""
    return values[:, :, 2:] - 2 * values[:, :, 1:-1] + values[:, :, :-2]


def velocity_error_mmps(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean L2 error between joint velocities, in mm/s."""
    pred = _velocity(prediction * _MM)
    tgt = _velocity(target * _MM)
    return (pred - tgt).norm(dim=-1).mean()


def acceleration_error_mmps2(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean L2 error between joint accelerations, in mm/s^2."""
    pred = _acceleration(prediction * _MM)
    tgt = _acceleration(target * _MM)
    return (pred - tgt).norm(dim=-1).mean()


def endpoint_drift_mm(prediction: Tensor, target: Tensor) -> Tensor:
    """Relative displacement error: compare last-minus-first for both sequences.

    ``prediction`` and ``target`` are ``[B,2,T,J,3]`` in metres; the result is
    the mean L2 error of the predicted displacement versus the target
    displacement, in millimetres.
    """
    pred_drift = prediction[:, :, -1] - prediction[:, :, 0]
    tgt_drift = target[:, :, -1] - target[:, :, 0]
    return (pred_drift - tgt_drift).norm(dim=-1).mean() * _MM
