"""Sample an SDF onto a voxel grid and (optionally) extract a surface mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hpe.geometry.implicit.sdf import SDF


@dataclass
class ScalarField:
    """A sampled SDF over a regular grid.

    Attributes
    ----------
    values : np.ndarray
        3D array of signed distances, shape ``(nx, ny, nz)``.
    origin : tuple[float, float, float]
        World coordinate of voxel ``[0,0,0]``.
    spacing : tuple[float, float, float]
        Voxel size along each axis [m].
    """

    values: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    @property
    def voxel_volume(self) -> float:
        sx, sy, sz = self.spacing
        return float(sx * sy * sz)

    @property
    def inside_count(self) -> int:
        return int(np.count_nonzero(self.values < 0.0))

    @property
    def solid_volume(self) -> float:
        """Approximate enclosed volume [m³] (inside-voxel count × voxel volume)."""
        return self.inside_count * self.voxel_volume

    def stats(self) -> dict:
        return {
            "grid_shape": list(self.values.shape),
            "spacing_mm": [round(s * 1000, 3) for s in self.spacing],
            "inside_voxels": self.inside_count,
            "solid_volume_m3": round(self.solid_volume, 9),
            "solid_volume_cm3": round(self.solid_volume * 1e6, 3),
        }


def sample(
    sdf: SDF,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    resolution: int = 48,
) -> ScalarField:
    """Sample ``sdf`` over an axis-aligned box ``bounds`` at ``resolution`` per axis."""
    (x0, x1), (y0, y1), (z0, z1) = bounds
    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    zs = np.linspace(z0, z1, resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    values = sdf(X, Y, Z)
    spacing = (
        (x1 - x0) / (resolution - 1),
        (y1 - y0) / (resolution - 1),
        (z1 - z0) / (resolution - 1),
    )
    return ScalarField(values=values, origin=(x0, y0, z0), spacing=spacing)


def surface_mesh(field: ScalarField):
    """Extract a triangle mesh via marching cubes, or ``None`` if scikit-image is absent.

    Returns ``(vertices, faces)`` in world coordinates [m], or ``None``.
    """
    try:
        from skimage import measure
    except Exception:  # noqa: BLE001 — optional dependency
        return None

    try:
        verts, faces, _normals, _vals = measure.marching_cubes(field.values, level=0.0)
    except (ValueError, RuntimeError):
        return None  # no surface crossing level 0 (empty or fully solid)

    # Voxel index → world coordinates.
    verts = verts * np.array(field.spacing) + np.array(field.origin)
    return verts, faces


def write_stl(vertices: np.ndarray, faces: np.ndarray, path: str) -> str:
    """Write an ASCII STL (no external deps)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["solid hpe_implicit"]
    for tri in faces:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 0 else n
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for v in (a, b, c):
            lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid hpe_implicit")
    p.write_text("\n".join(lines))
    return str(p)
