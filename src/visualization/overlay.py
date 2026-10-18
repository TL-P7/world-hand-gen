"""Array-only pinhole mesh overlay with a shared two-hand depth buffer."""

import numpy as np


def render_mesh_overlay(image, vertices_camera, faces, intrinsics, *, opacity=0.65):
    """Return RGB uint8 overlay; inputs are [H,W,3], [2,V,3], faces, [3,3].

    Uses perspective-correct depth for hand/hand and self occlusion. No scene
    depth is supplied, so real objects cannot occlude the generated mesh.
    Triangles crossing the near plane are discarded rather than extrapolated.
    """
    image = np.asarray(image)
    vertices = np.asarray(vertices_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be RGB uint8 [H,W,3]")
    if vertices.ndim != 3 or vertices.shape[0] != 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices_camera must be [2,V,3]")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be finite [3,3]")
    if not np.isfinite(vertices).all() or not 0 <= opacity <= 1:
        raise ValueError("vertices must be finite and opacity must lie in [0,1]")
    if len(faces) != 2:
        raise ValueError("provide one face array per hand")
    height, width = image.shape[:2]
    depth_buffer = np.full((height, width), np.inf)
    surface = np.zeros((height, width, 3))
    colors = np.array([[45, 160, 255], [255, 105, 65]])
    light = np.array([0.2, -0.4, -1.0])
    light /= np.linalg.norm(light)
    for hand in range(2):
        triangles = np.asarray(faces[hand])
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
            raise ValueError("faces must be integer [F,3]")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= vertices.shape[1]):
            raise ValueError("face index outside vertex array")
        points = vertices[hand]
        projected = points @ intrinsics.T
        for face in triangles:
            xyz = points[face]
            z = xyz[:, 2]
            if np.any(z <= 1e-4):
                continue
            uv = projected[face, :2] / projected[face, 2:3]
            lo = np.maximum(np.floor(uv.min(axis=0)), [0, 0]).astype(int)
            hi = np.minimum(np.ceil(uv.max(axis=0)), [width - 1, height - 1]).astype(int)
            if np.any(hi < lo):
                continue
            x0, y0 = lo
            x1, y1 = hi
            x, y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            a, b, c = uv
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-10:
                continue
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
            inv_depth = w0 / z[0] + w1 / z[1] + w2 / z[2]
            depth = 1 / np.maximum(inv_depth, 1e-12)
            region = depth_buffer[y0:y1 + 1, x0:x1 + 1]
            visible = inside & (depth < region)
            normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            normal /= max(np.linalg.norm(normal), 1e-12)
            shade = 0.45 + 0.55 * abs(normal @ light)
            region[visible] = depth[visible]
            surface[y0:y1 + 1, x0:x1 + 1][visible] = colors[hand] * shade
    mask = np.isfinite(depth_buffer)
    result = image.copy()
    result[mask] = np.clip((1 - opacity) * image[mask] + opacity * surface[mask], 0, 255).astype(np.uint8)
    return result
