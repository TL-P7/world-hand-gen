"""Matplotlib plots for already-prepared trajectories and MANO meshes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .geometry import _batched_intrinsics, _batched_vertices, project_points

_HAND_COLORS = ("#1f77b4", "#d62728")


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "visualization plotting requires matplotlib; install the optional plotting dependency"
        ) from error
    return plt


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _unbatch_joints(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim == 5:
        if result.shape[0] != 1:
            raise ValueError(f"{name} only accepts one batch item")
        result = result[0]
    if result.ndim != 4 or result.shape[0] != 2 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [2,T,J,3] or [1,2,T,J,3]")
    return result


def _unbatch_camera(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim == 3:
        if result.shape[0] != 1:
            raise ValueError(f"{name} only accepts one batch item")
        result = result[0]
    if result.ndim != 2 or result.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T,3] or [1,T,3]")
    return result


def _set_equal_3d(axis, values: Sequence[np.ndarray]) -> None:
    points = np.concatenate([value.reshape(-1, 3) for value in values if value.size], axis=0)
    if not points.size:
        return
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_trajectory_comparison(
    prediction_joints: Any,
    target_joints: Any,
    prediction_camera_translation: Any,
    target_camera_translation: Any,
    *,
    metrics: Mapping[str, float | None] | None = None,
    title: str | None = None,
    hand_names: Sequence[str] = ("left", "right"),
):
    """Plot predicted/target wrist and camera translation trajectories.

    All coordinates are plotted in the coordinate system supplied by the
    caller. The function performs no alignment, metric calculation, or data
    access. It returns a figure and a two-element tuple of axes.
    """
    prediction = _unbatch_joints(prediction_joints, "prediction_joints")
    target = _unbatch_joints(target_joints, "target_joints")
    prediction_camera = _unbatch_camera(
        prediction_camera_translation, "prediction_camera_translation"
    )
    target_camera = _unbatch_camera(target_camera_translation, "target_camera_translation")
    if prediction.shape != target.shape:
        raise ValueError("prediction_joints and target_joints must have identical shapes")
    if prediction.shape[1] != prediction_camera.shape[0] or target_camera.shape != prediction_camera.shape:
        raise ValueError("joint and camera trajectories must have the same time length")
    if len(hand_names) != 2:
        raise ValueError("hand_names must contain two names")

    plt = _plt()
    figure = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    hand_axis = figure.add_subplot(1, 2, 1, projection="3d")
    camera_axis = figure.add_subplot(1, 2, 2, projection="3d")
    for hand_index, hand_name in enumerate(hand_names):
        color = _HAND_COLORS[hand_index]
        hand_axis.plot(
            prediction[hand_index, :, 0, 0],
            prediction[hand_index, :, 0, 1],
            prediction[hand_index, :, 0, 2],
            color=color,
            linewidth=2,
            label=f"pred {hand_name}",
        )
        hand_axis.plot(
            target[hand_index, :, 0, 0],
            target[hand_index, :, 0, 1],
            target[hand_index, :, 0, 2],
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"GT {hand_name}",
        )
    camera_axis.plot(
        prediction_camera[:, 0], prediction_camera[:, 1], prediction_camera[:, 2],
        color="#2ca02c", linewidth=2, label="pred camera",
    )
    camera_axis.plot(
        target_camera[:, 0], target_camera[:, 1], target_camera[:, 2],
        color="#9467bd", linestyle="--", linewidth=1.4, label="GT camera",
    )
    hand_axis.set_title("Hand wrist trajectory")
    camera_axis.set_title("Camera translation trajectory")
    for axis in (hand_axis, camera_axis):
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.legend(fontsize=8)
    _set_equal_3d(
        hand_axis,
        [prediction[:, :, 0], target[:, :, 0]],
    )
    _set_equal_3d(camera_axis, [prediction_camera, target_camera])
    if title:
        figure.suptitle(title, fontsize=12)
    if metrics:
        text = []
        for name, value in metrics.items():
            text.append(f"{name}: {value:.2f}" if value is not None else f"{name}: n/a")
        figure.text(0.5, 0.005, "  |  ".join(text), ha="center", va="bottom", fontsize=8)
    return figure, (hand_axis, camera_axis)


def _mesh_faces(faces: Any) -> tuple[np.ndarray, ...]:
    values = faces if isinstance(faces, (tuple, list)) else (faces,)
    result = []
    for index, value in enumerate(values):
        array = _array(value, f"faces[{index}]")
        if array.ndim != 2 or array.shape[1] != 3 or not np.issubdtype(array.dtype, np.integer):
            # Float faces are accepted only when they are exactly integral,
            # which is convenient for tensors loaded from generic containers.
            if array.ndim != 2 or array.shape[1] != 3 or not np.equal(array, array.astype(int)).all():
                raise ValueError("faces must have shape [F,3] and integer indices")
            array = array.astype(int)
        result.append(array.astype(int, copy=False))
    return tuple(result)


def plot_projected_mesh_sequence(
    vertices_camera: Any,
    faces: Any,
    intrinsics: Any,
    *,
    frame_indices: Sequence[int] | None = None,
    image_size: tuple[int, int] = (384, 384),
    title: str | None = None,
    hand_names: Sequence[str] = ("left", "right"),
):
    """Plot selected generated MANO meshes after camera-frame projection.

    ``vertices_camera`` must already be in each frame's camera coordinate
    system. The function uses ``intrinsics`` only for projection and does not
    load images or calibration metadata.
    """
    vertices = _batched_vertices(vertices_camera)
    if vertices.shape[0] != 1:
        raise ValueError("plot_projected_mesh_sequence only accepts one batch item")
    vertices = vertices[0]
    face_arrays = _mesh_faces(faces)
    if len(face_arrays) == 1:
        face_arrays = face_arrays * vertices.shape[0]
    if len(face_arrays) != vertices.shape[0]:
        raise ValueError("faces must contain one mesh topology or one topology per hand")
    if len(hand_names) != vertices.shape[0]:
        raise ValueError("hand_names must match the number of hand streams")
    if frame_indices is None:
        count = min(vertices.shape[1], 6)
        frame_indices = np.linspace(0, vertices.shape[1] - 1, count, dtype=int).tolist()
    frame_indices = [int(index) for index in frame_indices]
    if not frame_indices or min(frame_indices) < 0 or max(frame_indices) >= vertices.shape[1]:
        raise ValueError("frame_indices must refer to valid sequence frames")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive width and height")

    uv, _, valid = project_points(vertices.unsqueeze(0), intrinsics)
    uv = uv[0].cpu().numpy()
    valid = valid[0].cpu().numpy()
    plt = _plt()
    figure, axes = plt.subplots(
        len(hand_names), len(frame_indices),
        figsize=(3.0 * len(frame_indices), 3.0 * len(hand_names)),
        squeeze=False,
        constrained_layout=True,
    )
    for hand_index, hand_name in enumerate(hand_names):
        for column, frame_index in enumerate(frame_indices):
            axis = axes[hand_index, column]
            polygons = []
            for face in face_arrays[hand_index]:
                if valid[hand_index, frame_index, face].all():
                    polygons.append(uv[hand_index, frame_index, face])
            if polygons:
                from matplotlib.collections import PolyCollection

                axis.add_collection(
                    PolyCollection(
                        polygons,
                        facecolors=_HAND_COLORS[hand_index],
                        edgecolors="black",
                        linewidths=0.15,
                        alpha=0.55,
                    )
                )
            axis.set_xlim(0, image_size[0])
            axis.set_ylim(image_size[1], 0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{hand_name} frame {frame_index}")
            axis.set_xlabel("u")
            axis.set_ylabel("v")
    if title:
        figure.suptitle(title, fontsize=12)
    return figure, axes
