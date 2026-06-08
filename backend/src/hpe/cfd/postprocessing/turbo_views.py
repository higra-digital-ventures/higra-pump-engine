"""Vistas turbo-especializadas: blade-to-blade + meridional de campo CFD.

Fase 18.4 — extrai fatias topologicamente relevantes do campo CFD:
  - Meridional average: média circunferencial em (r, z)
  - Blade-to-blade: corte em r = const, coord (θ, z)
  - Hub/shroud/midspan slices

Equivalente às vistas Turbo do CFX-Post / Fluent.

Usage
-----
    from hpe.cfd.postprocessing.turbo_views import (
        extract_meridional_average, extract_blade_to_blade,
    )

    mer = extract_meridional_average(grid_data, "U")
    print(mer.n_r, mer.n_z)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False
    np = None  # type: ignore


@dataclass
class MeridionalSlice:
    """Vista meridional (média circunferencial) em (r, z)."""
    field_name: str
    r: list[float]
    z: list[float]
    values: list[list[float]]   # values[i_z][i_r]
    n_r: int
    n_z: int
    min_value: float
    max_value: float

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "r": self.r, "z": self.z,
            "values": self.values,
            "n_r": self.n_r, "n_z": self.n_z,
            "min_value": round(self.min_value, 4),
            "max_value": round(self.max_value, 4),
        }


@dataclass
class BladeToBladeSlice:
    """Vista blade-to-blade em (θ, z) para um r específico."""
    field_name: str
    r_slice: float
    theta: list[float]
    z: list[float]
    values: list[list[float]]   # values[i_z][i_theta]
    min_value: float
    max_value: float

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "r_slice": round(self.r_slice, 4),
            "theta": self.theta, "z": self.z,
            "values": self.values,
            "min_value": round(self.min_value, 4),
            "max_value": round(self.max_value, 4),
        }


# ---------------------------------------------------------------------------
# Extrators
# ---------------------------------------------------------------------------

def extract_meridional_average(
    grid_data: dict,
    field_name: str = "U",
    n_r_bins: int = 30,
    n_z_bins: int = 20,
) -> MeridionalSlice:
    """Extrair vista meridional média circunferencial.

    Para cada (r, z), calcula a média do campo sobre todos os θ.
    Equivalente ao "meridional view" do CFX-Post.
    """
    if not _NP:
        return MeridionalSlice(field_name, [], [], [], 0, 0, 0.0, 0.0)

    grid = tuple(grid_data.get("grid", [1, 1, 1]))
    nx, ny, nz = grid
    field = np.array(grid_data["fields"].get(field_name, []))
    if field.size != nx * ny * nz:
        return MeridionalSlice(field_name, [], [], [], 0, 0, 0.0, 0.0)

    F = field.reshape((nz, ny, nx))
    bb = grid_data.get("bounding_box", {})
    x_min, y_min, z_min = bb.get("min", [0, 0, 0])
    x_max, y_max, z_max = bb.get("max", [1, 1, 1])

    # Coord radiais e axiais dos pontos do grid
    xv = np.linspace(x_min, x_max, nx)
    yv = np.linspace(y_min, y_max, ny)
    zv = np.linspace(z_min, z_max, nz)

    r_all = []
    z_all = []
    v_all = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                rr = math.hypot(xv[i], yv[j])
                r_all.append(rr)
                z_all.append(zv[k])
                v_all.append(float(F[k, j, i]))

    r_arr = np.array(r_all)
    z_arr = np.array(z_all)
    v_arr = np.array(v_all)

    r_edges = np.linspace(r_arr.min(), r_arr.max(), n_r_bins + 1)
    z_edges = np.linspace(z_arr.min(), z_arr.max(), n_z_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    means = np.zeros((n_z_bins, n_r_bins))
    counts = np.zeros((n_z_bins, n_r_bins))

    r_idx = np.clip(np.digitize(r_arr, r_edges) - 1, 0, n_r_bins - 1)
    z_idx = np.clip(np.digitize(z_arr, z_edges) - 1, 0, n_z_bins - 1)

    for p in range(len(v_arr)):
        means[z_idx[p], r_idx[p]] += v_arr[p]
        counts[z_idx[p], r_idx[p]] += 1

    counts[counts == 0] = 1
    means /= counts

    return MeridionalSlice(
        field_name=field_name,
        r=[round(x, 5) for x in r_centers.tolist()],
        z=[round(x, 5) for x in z_centers.tolist()],
        values=[[round(v, 4) for v in row] for row in means.tolist()],
        n_r=n_r_bins, n_z=n_z_bins,
        min_value=float(means.min()),
        max_value=float(means.max()),
    )


def extract_blade_to_blade(
    grid_data: dict,
    field_name: str = "U",
    r_slice: Optional[float] = None,
    n_theta_bins: int = 36,
    n_z_bins: int = 20,
    r_tolerance: float = 0.02,
) -> BladeToBladeSlice:
    """Extrair vista blade-to-blade em um raio específico.

    Projeta o campo em coord (θ, z) para todos os pontos com r ≈ r_slice.
    Se r_slice for None, usa a média do grid.
    """
    if not _NP:
        return BladeToBladeSlice(field_name, 0.0, [], [], [], 0.0, 0.0)

    grid = tuple(grid_data.get("grid", [1, 1, 1]))
    nx, ny, nz = grid
    field = np.array(grid_data["fields"].get(field_name, []))
    if field.size != nx * ny * nz:
        return BladeToBladeSlice(field_name, 0.0, [], [], [], 0.0, 0.0)

    F = field.reshape((nz, ny, nx))
    bb = grid_data.get("bounding_box", {})
    x_min, y_min, z_min = bb.get("min", [0, 0, 0])
    x_max, y_max, z_max = bb.get("max", [1, 1, 1])

    xv = np.linspace(x_min, x_max, nx)
    yv = np.linspace(y_min, y_max, ny)
    zv = np.linspace(z_min, z_max, nz)

    r_grid = np.sqrt(
        xv[None, None, :] ** 2 + yv[None, :, None] ** 2
    ).repeat(nz, axis=0)
    r_grid = np.broadcast_to(r_grid, F.shape)

    r_max_grid = float(r_grid.max())
    if r_slice is None:
        r_slice = 0.75 * r_max_grid   # default: 75% do raio

    mask = np.abs(r_grid - r_slice) < r_tolerance * r_max_grid

    theta_pts = []
    z_pts = []
    v_pts = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not mask[k, j, i]:
                    continue
                theta = math.atan2(yv[j], xv[i])
                theta_pts.append(theta)
                z_pts.append(zv[k])
                v_pts.append(float(F[k, j, i]))

    if not v_pts:
        return BladeToBladeSlice(field_name, r_slice, [], [], [], 0.0, 0.0)

    theta_arr = np.array(theta_pts)
    z_arr = np.array(z_pts)
    v_arr = np.array(v_pts)

    th_edges = np.linspace(-math.pi, math.pi, n_theta_bins + 1)
    z_edges = np.linspace(z_arr.min(), z_arr.max(), n_z_bins + 1)

    grid_out = np.zeros((n_z_bins, n_theta_bins))
    counts = np.zeros_like(grid_out)

    th_idx = np.clip(np.digitize(theta_arr, th_edges) - 1, 0, n_theta_bins - 1)
    z_idx = np.clip(np.digitize(z_arr, z_edges) - 1, 0, n_z_bins - 1)

    for p in range(len(v_arr)):
        grid_out[z_idx[p], th_idx[p]] += v_arr[p]
        counts[z_idx[p], th_idx[p]] += 1

    counts[counts == 0] = 1
    grid_out /= counts

    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return BladeToBladeSlice(
        field_name=field_name,
        r_slice=float(r_slice),
        theta=[round(x, 4) for x in th_centers.tolist()],
        z=[round(x, 4) for x in z_centers.tolist()],
        values=[[round(v, 4) for v in row] for row in grid_out.tolist()],
        min_value=float(grid_out.min()),
        max_value=float(grid_out.max()),
    )
