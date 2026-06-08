"""Signed-distance-field (SDF) primitives and boolean operations (pure numpy).

An SDF is a callable ``f(x, y, z) -> distance`` where the arguments are numpy
arrays of identical shape and the result is negative inside the solid, zero on
the surface, positive outside. Primitives and operators compose into arbitrarily
complex geometry without any B-rep kernel.

Distances are in metres (consistent with the rest of HPE).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SDF = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def sphere(radius: float, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SDF:
    cx, cy, cz = center

    def f(x, y, z):
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius

    return f


def capped_cylinder_z(
    radius: float,
    z0: float,
    z1: float,
    center_xy: tuple[float, float] = (0.0, 0.0),
) -> SDF:
    """Finite cylinder aligned with the z-axis, capped at ``z0`` and ``z1``."""
    cx, cy = center_xy
    zc = 0.5 * (z0 + z1)
    half_h = 0.5 * abs(z1 - z0)

    def f(x, y, z):
        d_radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
        d_axial = np.abs(z - zc) - half_h
        outside = np.sqrt(np.maximum(d_radial, 0.0) ** 2 + np.maximum(d_axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
        return outside + inside

    return f


def box(
    half_extents: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SDF:
    """Axis-aligned box with the given half-extents."""
    hx, hy, hz = half_extents
    cx, cy, cz = center

    def f(x, y, z):
        qx = np.abs(x - cx) - hx
        qy = np.abs(y - cy) - hy
        qz = np.abs(z - cz) - hz
        outside = np.sqrt(
            np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2
        )
        inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
        return outside + inside

    return f


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def rotate_z(inner: SDF, angle_rad: float) -> SDF:
    """Rotate an SDF about the z-axis by ``angle_rad`` (query-point transform)."""
    c, s = np.cos(-angle_rad), np.sin(-angle_rad)

    def f(x, y, z):
        xr = c * x - s * y
        yr = s * x + c * y
        return inner(xr, yr, z)

    return f


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------

def union(*sdfs: SDF) -> SDF:
    def f(x, y, z):
        out = sdfs[0](x, y, z)
        for s in sdfs[1:]:
            out = np.minimum(out, s(x, y, z))
        return out

    return f


def intersection(*sdfs: SDF) -> SDF:
    def f(x, y, z):
        out = sdfs[0](x, y, z)
        for s in sdfs[1:]:
            out = np.maximum(out, s(x, y, z))
        return out

    return f


def subtract(a: SDF, b: SDF) -> SDF:
    """Solid ``a`` with ``b`` removed."""
    def f(x, y, z):
        return np.maximum(a(x, y, z), -b(x, y, z))

    return f


def smooth_union(a: SDF, b: SDF, k: float = 0.05) -> SDF:
    """Blended union with smoothing radius ``k`` (metres)."""
    def f(x, y, z):
        da, db = a(x, y, z), b(x, y, z)
        h = np.clip(0.5 + 0.5 * (db - da) / k, 0.0, 1.0)
        return db * (1 - h) + da * h - k * h * (1.0 - h)

    return f
