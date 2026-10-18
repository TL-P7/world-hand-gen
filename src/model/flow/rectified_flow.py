"""Rectified-flow interpolation and velocity math.

Both ``ManoFlowState`` and ``CameraFlowState`` support the same per-field
operations, so a single pair of functions handles both.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Union

from torch import Tensor

from .state import CameraFlowState, ManoFlowState

FlowState = Union[ManoFlowState, CameraFlowState]


def _interpolate_field(source: Tensor, destination: Tensor, t: Tensor) -> Tensor:
    t = t.reshape((-1,) + (1,) * (source.ndim - 1))
    return (1 - t) * source + t * destination


def flow_interpolate(noise: FlowState, target: FlowState, flow_time: Tensor) -> FlowState:
    """Linear path ``X_t = (1-t) * noise + t * target``."""
    return type(noise)(
        **{
            field.name: _interpolate_field(
                getattr(noise, field.name), getattr(target, field.name), flow_time
            )
            for field in fields(noise)
        }
    )


def flow_velocity_from_x0(state: FlowState, x0: FlowState, flow_time: Tensor) -> FlowState:
    """Rectified-flow velocity ``v = (x0 - x_t) / (1 - t)``."""
    if (flow_time >= 1).any():
        raise ValueError("flow_time must be strictly less than 1 for velocity")
    velocity = {}
    for field in fields(state):
        current = getattr(state, field.name)
        clean = getattr(x0, field.name)
        denominator = (1 - flow_time).reshape((-1,) + (1,) * (current.ndim - 1))
        velocity[field.name] = (clean - current) / denominator
    return type(state)(**velocity)
