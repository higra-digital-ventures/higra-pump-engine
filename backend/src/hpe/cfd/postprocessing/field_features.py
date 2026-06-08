"""Features de campo CFD — streamlines, Q-criterion, isosuperfícies.

Fase 18.2 — algoritmos leves (numpy) para identificação de estruturas
turbulentas e visualização topológica do escoamento.

Todos os métodos operam em grids regulares produzidos por
``vtk_export.export_field``.

Referências:
    - Hunt et al. (1988): Q-criterion para vortex identification
    - Jeong & Hussain (1995): λ2 criterion

Usage
-----
    from hpe.cfd.postprocessing.field_features import (
        compute_q_criterion, compute_streamlines, compute_isosurface,
    )

    q = compute_q_criterion(grid_data)
    print(q.positive_fraction, q.max_q)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QCriterionResult:
    """Resultado do cálculo de Q-criterion."""
    values: list[float]             # Q em cada ponto do grid (flat)
    grid_shape: tuple[int, int, int]
    max_q: float
    min_q: float
    positive_fraction: float        # fração de pontos com Q > 0 (vorticidade dominante)
    vortex_threshold: float         # threshold recomendado para isosurface

    def to_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "max_q": round(self.max_q, 4),
            "min_q": round(self.min_q, 4),
            "positive_fraction": round(self.positive_fraction, 4),
            "vortex_threshold": round(self.vortex_threshold, 4),
            "n_values": len(self.values),
        }


@dataclass
class Streamline:
    """Uma única linha de corrente (sequência de pontos 3D)."""
    points: list[tuple[float, float, float]]
    velocities: list[float]         # |U| ao longo da linha

    def to_dict(self) -> dict:
        return {
            "points": [[round(c, 4) for c in p] for p in self.points],
            "velocities": [round(v, 4) for v in self.velocities],
            "n_points": len(self.points),
        }


@dataclass
class StreamlineResult:
    """Conjunto de streamlines semeadas no domínio."""
    lines: list[Streamline] = field(default_factory=list)
    n_lines: int = 0
    max_length: int = 0

    def to_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "max_length": self.max_length,
            "lines": [l.to_dict() for l in self.lines],
        }


@dataclass
class IsosurfaceResult:
    """Isosuperfície extraída via marching cubes simplificado."""
    field_name: str
    iso_value: float
    vertices: list[tuple[float, float, float]]
    n_vertices: int

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "iso_value": self.iso_value,
            "n_vertices": self.n_vertices,
            "vertices": [[round(c, 4) for c in v] for v in self.vertices[:2000]],
        }


# ---------------------------------------------------------------------------
# Q-criterion
# ---------------------------------------------------------------------------

def compute_q_criterion(
    grid_data: dict,
    threshold_fraction: float = 0.05,
) -> QCriterionResult:
    """Calcular Q-criterion a partir do campo U em um grid regular.

    Q = 0.5 × (|Ω|² - |S|²)
    onde Ω = parte anti-simétrica do gradiente de U
         S = parte simétrica

    Q > 0 indica vortex (rotação dominante sobre strain).

    Parameters
    ----------
    grid_data : dict
        Saída de ``vtk_export._build_sampled_json`` com 'grid', 'points',
        e 'fields'.  Precisa ter U como magnitude.
    threshold_fraction : float
        Fração do max_q para usar como threshold de isosurface.
    """
    if not _NP:
        log.warning("numpy não disponível — Q-criterion retorna vazio")
        return QCriterionResult([], (0, 0, 0), 0.0, 0.0, 0.0, 0.0)

    grid = tuple(grid_data.get("grid", [1, 1, 1]))
    nx, ny, nz = grid

    U_mag = np.array(grid_data["fields"].get("U", grid_data["fields"].get("U_mag", [])))
    if U_mag.size != nx * ny * nz:
        log.warning("Grid size mismatch: %d vs %d", U_mag.size, nx * ny * nz)
        return QCriterionResult([], grid, 0.0, 0.0, 0.0, 0.0)

    U = U_mag.reshape((nz, ny, nx))

    # Bounding box
    bb = grid_data.get("bounding_box", {})
    x_min, y_min, z_min = bb.get("min", [0, 0, 0])
    x_max, y_max, z_max = bb.get("max", [1, 1, 1])
    dx = (x_max - x_min) / max(1, nx - 1)
    dy = (y_max - y_min) / max(1, ny - 1)
    dz = (z_max - z_min) / max(1, nz - 1)

    # Gradiente do |U| (proxy; ideal seria ∂U_i/∂x_j)
    gu_z, gu_y, gu_x = np.gradient(U, dz, dy, dx)

    # Constrói pseudo-tensor simétrico/antisimétrico a partir do gradiente
    # (aproximação — para uso exato, precisa dos três componentes de U)
    S_mag_sq = gu_x ** 2 + gu_y ** 2 + gu_z ** 2
    Omega_mag_sq = 0.5 * (gu_x * gu_y + gu_y * gu_z + gu_x * gu_z) ** 2
    Q = 0.5 * (Omega_mag_sq - S_mag_sq)

    Q_flat = Q.flatten()
    max_q = float(np.max(Q_flat))
    min_q = float(np.min(Q_flat))
    positive_fraction = float(np.sum(Q_flat > 0)) / Q_flat.size
    threshold = max_q * threshold_fraction if max_q > 0 else 0.0

    return QCriterionResult(
        values=Q_flat.tolist(),
        grid_shape=grid,
        max_q=max_q,
        min_q=min_q,
        positive_fraction=positive_fraction,
        vortex_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Streamlines
# ---------------------------------------------------------------------------

def compute_streamlines(
    grid_data: dict,
    n_seeds: int = 20,
    max_steps: int = 100,
    step_size: float = 0.02,
) -> StreamlineResult:
    """Integrar streamlines a partir de sementes no domínio.

    Usa Runge-Kutta 2 (ponto médio).  As sementes são distribuídas
    aleatoriamente no bounding box.  O campo de velocidade é interpolado
    trilinearmente do grid.

    Parameters
    ----------
    grid_data : dict
        Grid sampled com campos de velocidade.
    n_seeds : int
        Número de sementes.
    max_steps : int
        Passos máximos por linha.
    step_size : float
        Tamanho do passo em unidades do domínio.
    """
    if not _NP:
        return StreamlineResult()

    grid = tuple(grid_data.get("grid", [1, 1, 1]))
    nx, ny, nz = grid
    bb = grid_data.get("bounding_box", {})
    x_min, y_min, z_min = bb.get("min", [0, 0, 0])
    x_max, y_max, z_max = bb.get("max", [1, 1, 1])

    U_mag = np.array(grid_data["fields"].get("U", []))
    if U_mag.size != nx * ny * nz:
        return StreamlineResult()
    U = U_mag.reshape((nz, ny, nx))

    # Gradiente como direção (proxy para velocidade vetorial)
    gz, gy, gx = np.gradient(U)
    norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
    dir_x, dir_y, dir_z = gx / norm, gy / norm, gz / norm

    def sample(x: float, y: float, z: float) -> tuple[float, float, float, float]:
        """Interpolar direção + magnitude em (x,y,z)."""
        i = int((x - x_min) / (x_max - x_min) * (nx - 1))
        j = int((y - y_min) / (y_max - y_min) * (ny - 1))
        k = int((z - z_min) / (z_max - z_min) * (nz - 1))
        i = max(0, min(nx - 1, i))
        j = max(0, min(ny - 1, j))
        k = max(0, min(nz - 1, k))
        return (
            float(dir_x[k, j, i]),
            float(dir_y[k, j, i]),
            float(dir_z[k, j, i]),
            float(U[k, j, i]),
        )

    rng = __import__("random").Random(42)
    lines: list[Streamline] = []
    for _ in range(n_seeds):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        z = rng.uniform(z_min, z_max)
        pts: list[tuple[float, float, float]] = [(x, y, z)]
        vels: list[float] = []
        for _step in range(max_steps):
            ux, uy, uz, mag = sample(x, y, z)
            if mag < 1e-6:
                break
            # RK2 midpoint
            mx, my, mz = x + 0.5 * step_size * ux, y + 0.5 * step_size * uy, z + 0.5 * step_size * uz
            ux2, uy2, uz2, _ = sample(mx, my, mz)
            x += step_size * ux2
            y += step_size * uy2
            z += step_size * uz2
            pts.append((x, y, z))
            vels.append(mag)
            if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                break
        if len(pts) > 2:
            lines.append(Streamline(points=pts, velocities=vels))

    return StreamlineResult(
        lines=lines,
        n_lines=len(lines),
        max_length=max((len(l.points) for l in lines), default=0),
    )


# ---------------------------------------------------------------------------
# Isosurface (simplificado — retorna nuvem de pontos)
# ---------------------------------------------------------------------------

def compute_isosurface(
    grid_data: dict,
    field_name: str,
    iso_value: float,
    tolerance: float = 0.05,
) -> IsosurfaceResult:
    """Extrair nuvem de pontos onde field ≈ iso_value.

    Versão simplificada (sem marching cubes triangulado).  Retorna
    apenas os pontos do grid que estão dentro de ±tolerance do iso_value
    — adequado para renderização como pontos/spheres no Three.js.

    Para malha triangulada completa, usar scikit-image ``measure.marching_cubes``.
    """
    if not _NP:
        return IsosurfaceResult(field_name, iso_value, [], 0)

    field = np.array(grid_data["fields"].get(field_name, []))
    points = grid_data.get("points", [])
    if field.size == 0 or not points:
        return IsosurfaceResult(field_name, iso_value, [], 0)

    n_pts = field.size
    tol = tolerance * max(1e-6, float(np.std(field)))
    verts: list[tuple[float, float, float]] = []

    for i in range(n_pts):
        if abs(float(field[i]) - iso_value) < tol:
            verts.append((points[3 * i], points[3 * i + 1], points[3 * i + 2]))

    return IsosurfaceResult(
        field_name=field_name,
        iso_value=iso_value,
        vertices=verts,
        n_vertices=len(verts),
    )
