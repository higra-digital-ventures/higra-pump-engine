"""Blade intersection / collision detection for impeller blades.

Checks for geometric intersections between adjacent blades (PS of blade[i]
vs SS of blade[i+1]) and between blades and hub/shroud surfaces.

Uses surface sampling on a (span, chord) grid and minimum-distance
computation to flag regions where clearance is below tolerance.

References:
    Gulich (2014) — Centrifugal Pumps, Ch. 7 (impeller geometry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProblemRegion:
    """A region where blade clearance is below tolerance."""

    span_fraction: float  # 0 = hub, 1 = shroud
    chord_fraction: float  # 0 = LE, 1 = TE
    clearance_mm: float  # Minimum clearance in this region [mm]
    blade_pair: tuple[int, int] = (0, 0)  # (blade_i, blade_j)


@dataclass
class CollisionReport:
    """Result of blade intersection analysis."""

    has_intersection: bool
    intersection_points: list[tuple[float, float, float]]  # (x, y, z) coords
    min_clearance_mm: float
    problem_regions: list[ProblemRegion]
    blade_hub_clearance_mm: float  # Minimum blade-hub clearance
    blade_shroud_clearance_mm: float  # Minimum blade-shroud clearance
    n_blades_checked: int
    summary: str


# ---------------------------------------------------------------------------
# Surface representation helpers
# ---------------------------------------------------------------------------

def _blade_surface_to_cartesian(
    blade_points: list[tuple[float, float]],
    z_hub: float,
    z_shroud: float,
    n_span: int = 20,
) -> np.ndarray:
    """Convert (r, theta) blade profile to 3D Cartesian grid.

    Creates a surface grid by extruding the 2D blade profile from
    hub to shroud along the z-axis.

    Args:
        blade_points: List of (r, theta) points along the blade.
        z_hub: Hub z-coordinate [m].
        z_shroud: Shroud z-coordinate [m].
        n_span: Number of spanwise stations.

    Returns:
        Array of shape (n_span, n_chord, 3) with (x, y, z) coordinates.
    """
    n_chord = len(blade_points)
    surface = np.zeros((n_span, n_chord, 3))

    z_values = np.linspace(z_hub, z_shroud, n_span)

    for i, z in enumerate(z_values):
        for j, (r, theta) in enumerate(blade_points):
            surface[i, j, 0] = r * math.cos(theta)
            surface[i, j, 1] = r * math.sin(theta)
            surface[i, j, 2] = z

    return surface


def _rotate_surface(
    surface: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate a blade surface around the z-axis.

    Args:
        surface: Shape (n_span, n_chord, 3).
        angle_rad: Rotation angle [rad].

    Returns:
        Rotated surface with the same shape.
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = surface.copy()
    x = surface[..., 0]
    y = surface[..., 1]
    rotated[..., 0] = x * cos_a - y * sin_a
    rotated[..., 1] = x * sin_a + y * cos_a

    return rotated


def _compute_min_distances(
    surface_a: np.ndarray,
    surface_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute minimum distance between two surface grids.

    For each point on surface_a, find the closest point on surface_b.

    Args:
        surface_a: Shape (n_span, n_chord, 3).
        surface_b: Shape (n_span, n_chord, 3).

    Returns:
        (distance_grid, global_min_distance) where distance_grid has
        shape (n_span_a, n_chord_a).
    """
    n_span_a, n_chord_a, _ = surface_a.shape
    n_span_b, n_chord_b, _ = surface_b.shape

    # Flatten surface_b for broadcasting
    pts_b = surface_b.reshape(-1, 3)  # (M, 3)

    distances = np.zeros((n_span_a, n_chord_a))

    for i in range(n_span_a):
        for j in range(n_chord_a):
            pt = surface_a[i, j]  # (3,)
            diffs = pts_b - pt  # (M, 3)
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            distances[i, j] = float(np.min(dists))

    global_min = float(np.min(distances))
    return distances, global_min


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_intersections(
    blade_surfaces: list[list[tuple[float, float]]],
    n_blades: int | None = None,
    tolerance_mm: float = 0.5,
    z_hub: float = 0.0,
    z_shroud: float = 0.05,
    n_span: int = 15,
    hub_radius: float | None = None,
    shroud_radius: float | None = None,
) -> CollisionReport:
    """Detect blade-to-blade and blade-to-wall intersections.

    Checks:
        1. Adjacent blade pairs: PS of blade[i] vs SS of blade[i+1]
        2. Blade-hub clearance (if hub_radius given)
        3. Blade-shroud clearance (if shroud_radius given)

    Args:
        blade_surfaces: List of blade profiles.  Each blade is a list
            of two sub-lists: ``[pressure_side_points, suction_side_points]``
            where each sub-list contains ``(r, theta)`` tuples.
            If a single profile is provided, it is replicated around
            the circumference using ``n_blades``.
        n_blades: Number of blades (used for replication).  If ``None``,
            inferred from ``len(blade_surfaces)``.
        tolerance_mm: Clearance threshold [mm].
        z_hub: Hub z-coordinate [m].
        z_shroud: Shroud z-coordinate [m].
        n_span: Number of spanwise sample stations.
        hub_radius: Hub surface radius [m] (for blade-hub check).
        shroud_radius: Shroud surface radius [m] (for blade-shroud check).

    Returns:
        :class:`CollisionReport` with full diagnostics.
    """
    if not blade_surfaces:
        return CollisionReport(
            has_intersection=False,
            intersection_points=[],
            min_clearance_mm=999.0,
            problem_regions=[],
            blade_hub_clearance_mm=999.0,
            blade_shroud_clearance_mm=999.0,
            n_blades_checked=0,
            summary="No blade surfaces provided.",
        )

    # Determine blade count and layout
    if n_blades is None:
        n_blades = len(blade_surfaces)

    blade_spacing_rad = 2.0 * math.pi / n_blades

    # Parse blade surfaces
    # Each element may be [ps_points, ss_points] or just a flat profile
    ps_surfaces_3d: list[np.ndarray] = []
    ss_surfaces_3d: list[np.ndarray] = []

    for blade_idx in range(n_blades):
        # Get the blade profile (replicate if only one provided)
        src_idx = blade_idx % len(blade_surfaces)
        blade_data = blade_surfaces[src_idx]

        if isinstance(blade_data, (list, tuple)) and len(blade_data) == 2:
            ps_pts = blade_data[0]
            ss_pts = blade_data[1]
        else:
            # Single profile — use as both PS and SS (thin blade approx)
            ps_pts = blade_data
            ss_pts = blade_data

        rotation = blade_idx * blade_spacing_rad

        ps_3d = _blade_surface_to_cartesian(ps_pts, z_hub, z_shroud, n_span)
        ss_3d = _blade_surface_to_cartesian(ss_pts, z_hub, z_shroud, n_span)

        ps_3d = _rotate_surface(ps_3d, rotation)
        ss_3d = _rotate_surface(ss_3d, rotation)

        ps_surfaces_3d.append(ps_3d)
        ss_surfaces_3d.append(ss_3d)

    # --- Check adjacent blade pairs ---
    problem_regions: list[ProblemRegion] = []
    intersection_points: list[tuple[float, float, float]] = []
    global_min_clearance = float("inf")
    tolerance_m = tolerance_mm / 1000.0

    for i in range(n_blades):
        j = (i + 1) % n_blades

        # PS of blade[i] vs SS of blade[j]
        dist_grid, min_dist = _compute_min_distances(
            ps_surfaces_3d[i],
            ss_surfaces_3d[j],
        )

        min_dist_mm = min_dist * 1000.0
        global_min_clearance = min(global_min_clearance, min_dist_mm)

        # Find problem regions
        n_s, n_c = dist_grid.shape
        for si in range(n_s):
            for ci in range(n_c):
                d_mm = dist_grid[si, ci] * 1000.0
                if d_mm < tolerance_mm:
                    span_frac = si / max(n_s - 1, 1)
                    chord_frac = ci / max(n_c - 1, 1)
                    problem_regions.append(ProblemRegion(
                        span_fraction=round(span_frac, 3),
                        chord_fraction=round(chord_frac, 3),
                        clearance_mm=round(d_mm, 3),
                        blade_pair=(i, j),
                    ))

                    if d_mm < 1e-3:  # Actual intersection
                        pt = ps_surfaces_3d[i][si, ci]
                        intersection_points.append((
                            round(float(pt[0]), 6),
                            round(float(pt[1]), 6),
                            round(float(pt[2]), 6),
                        ))

    # --- Check blade-hub and blade-shroud clearances ---
    blade_hub_min_mm = 999.0
    blade_shroud_min_mm = 999.0

    if hub_radius is not None:
        for i in range(n_blades):
            for surf in [ps_surfaces_3d[i], ss_surfaces_3d[i]]:
                # Hub is at z = z_hub; check radial clearance at hub span
                hub_row = surf[0, :, :]  # First spanwise row (hub)
                r_blade = np.sqrt(hub_row[:, 0] ** 2 + hub_row[:, 1] ** 2)
                clearances = (r_blade - hub_radius) * 1000.0
                min_c = float(np.min(np.abs(clearances)))
                blade_hub_min_mm = min(blade_hub_min_mm, min_c)

                if min_c < tolerance_mm:
                    for ci in range(len(clearances)):
                        if abs(clearances[ci]) < tolerance_mm:
                            problem_regions.append(ProblemRegion(
                                span_fraction=0.0,
                                chord_fraction=ci / max(len(clearances) - 1, 1),
                                clearance_mm=round(abs(float(clearances[ci])), 3),
                                blade_pair=(i, -1),  # -1 = hub
                            ))

    if shroud_radius is not None:
        for i in range(n_blades):
            for surf in [ps_surfaces_3d[i], ss_surfaces_3d[i]]:
                # Shroud is at z = z_shroud; check radial clearance
                shroud_row = surf[-1, :, :]  # Last spanwise row (shroud)
                r_blade = np.sqrt(shroud_row[:, 0] ** 2 + shroud_row[:, 1] ** 2)
                clearances = (shroud_radius - r_blade) * 1000.0
                min_c = float(np.min(np.abs(clearances)))
                blade_shroud_min_mm = min(blade_shroud_min_mm, min_c)

                if min_c < tolerance_mm:
                    for ci in range(len(clearances)):
                        if abs(clearances[ci]) < tolerance_mm:
                            problem_regions.append(ProblemRegion(
                                span_fraction=1.0,
                                chord_fraction=ci / max(len(clearances) - 1, 1),
                                clearance_mm=round(abs(float(clearances[ci])), 3),
                                blade_pair=(i, -2),  # -2 = shroud
                            ))

    has_intersection = len(intersection_points) > 0

    # Build summary
    n_problems = len(problem_regions)
    if has_intersection:
        summary = (
            f"INTERSECTION DETECTED: {len(intersection_points)} intersection "
            f"point(s) found. Min clearance: {global_min_clearance:.3f} mm."
        )
    elif n_problems > 0:
        summary = (
            f"WARNING: {n_problems} region(s) below {tolerance_mm} mm tolerance. "
            f"Min clearance: {global_min_clearance:.3f} mm."
        )
    else:
        summary = (
            f"OK: All clearances above {tolerance_mm} mm tolerance. "
            f"Min clearance: {global_min_clearance:.3f} mm."
        )

    return CollisionReport(
        has_intersection=has_intersection,
        intersection_points=intersection_points,
        min_clearance_mm=round(global_min_clearance, 3),
        problem_regions=problem_regions,
        blade_hub_clearance_mm=round(blade_hub_min_mm, 3),
        blade_shroud_clearance_mm=round(blade_shroud_min_mm, 3),
        n_blades_checked=n_blades,
        summary=summary,
    )
