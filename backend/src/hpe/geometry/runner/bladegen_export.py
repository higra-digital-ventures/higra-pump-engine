"""ANSYS BladeGen .bgd format writer.

Exports impeller geometry in the BladeGen text-based .bgd format
containing hub/shroud profiles, blade sections in cylindrical
coordinates, and thickness distributions.

Reference format:
    BLADE GENERATOR INPUT FILE
    VERSION 1.0
    UNITS MM
    HUB PROFILE / SHROUD PROFILE sections
    BLADE SECTION n  (spanwise sections)
    THICKNESS DISTRIBUTION
    NUMBER OF BLADES
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np


@dataclass
class BladePoint:
    """A 3D point in Cartesian coordinates."""

    x: float
    y: float
    z: float


def _cartesian_to_cylindrical(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """Convert Cartesian (x, y, z) to cylindrical (theta, r, z).

    Args:
        x: Cartesian x coordinate.
        y: Cartesian y coordinate.
        z: Axial coordinate (unchanged).

    Returns:
        Tuple of (theta_deg, r, z) where theta is in degrees.
    """
    r = math.sqrt(x * x + y * y)
    theta_rad = math.atan2(y, x)
    theta_deg = math.degrees(theta_rad)
    return theta_deg, r, z


def _format_profile(
    points: Sequence[Any], scale: float = 1000.0
) -> str:
    """Format a meridional profile (hub or shroud) as z r lines.

    Points are expected to have .x, .y, .z attributes (Cartesian)
    or be dicts with those keys.  The profile is expressed as (z, r)
    where r = sqrt(x^2 + y^2) and z = z, both scaled to mm.

    Args:
        points: Sequence of blade points (dicts or objects).
        scale: Multiplier to convert metres to mm (default 1000).

    Returns:
        Multi-line string with n_points header and z r rows.
    """
    lines: list[str] = []
    coords: list[Tuple[float, float]] = []

    for pt in points:
        if isinstance(pt, dict):
            px, py, pz = pt["x"], pt["y"], pt["z"]
        else:
            px, py, pz = pt.x, pt.y, pt.z
        r = math.sqrt(px * px + py * py) * scale
        z = pz * scale
        coords.append((z, r))

    lines.append(str(len(coords)))
    for z_val, r_val in coords:
        lines.append(f"{z_val:12.4f} {r_val:12.4f}")

    return "\n".join(lines)


def _format_blade_section(
    section_points: Sequence[Any],
    section_index: int,
    scale: float = 1000.0,
) -> str:
    """Format one spanwise blade section as THETA R Z lines.

    Args:
        section_points: Points along the blade camberline for this span.
        section_index: 0-based spanwise section index.
        scale: Multiplier to convert m to mm.

    Returns:
        Multi-line string with section header and coordinate rows.
    """
    lines: list[str] = [f"BLADE SECTION {section_index}"]
    rows: list[str] = []

    for pt in section_points:
        if isinstance(pt, dict):
            px, py, pz = pt["x"], pt["y"], pt["z"]
        else:
            px, py, pz = pt.x, pt.y, pt.z

        theta_deg, r, z = _cartesian_to_cylindrical(px, py, pz)
        rows.append(f"{theta_deg:12.4f} {r * scale:12.4f} {z * scale:12.4f}")

    lines.append(str(len(rows)))
    lines.extend(rows)
    return "\n".join(lines)


def _default_thickness_distribution(n_points: int = 11) -> str:
    """Generate a default NACA-like thickness distribution.

    Uses a parabolic distribution: thickest at ~30% chord,
    tapering to zero at LE and TE.

    Args:
        n_points: Number of points along normalised chord.

    Returns:
        Multi-line string with fraction and thickness_mm rows.
    """
    lines: list[str] = ["THICKNESS DISTRIBUTION", str(n_points)]
    fracs = np.linspace(0.0, 1.0, n_points)
    for f in fracs:
        # Parabolic distribution peaked at 30% chord, max ~3 mm
        t = 3.0 * 4.0 * f * (1.0 - f) * (1.0 if f <= 0.3 else (1.0 - f) / 0.7)
        t = max(t, 0.0)
        lines.append(f"{f:8.4f} {t:8.4f}")
    return "\n".join(lines)


def export_bladegen(
    sizing_result: Any,
    blade_surfaces: Sequence[Any],
    hub_profile: Sequence[Any],
    shroud_profile: Sequence[Any],
    blade_count: int,
) -> str:
    """Export impeller geometry in ANSYS BladeGen .bgd format.

    Args:
        sizing_result: SizingResult dataclass from meanline sizing.
        blade_surfaces: List of BladeSurface objects, each with .ps and .ss
            (lists of spanwise lists of BladePoint3D).
        hub_profile: List of 3D points defining the hub meridional profile.
        shroud_profile: List of 3D points defining the shroud meridional profile.
        blade_count: Number of main blades.

    Returns:
        Complete .bgd file content as a string.
    """
    sections: list[str] = []

    # Header
    sections.append("BLADE GENERATOR INPUT FILE")
    sections.append("VERSION 1.0")
    sections.append("UNITS MM")
    sections.append("")

    # Hub profile
    sections.append("HUB PROFILE")
    sections.append(_format_profile(hub_profile))
    sections.append("")

    # Shroud profile
    sections.append("SHROUD PROFILE")
    sections.append(_format_profile(shroud_profile))
    sections.append("")

    # Blade sections — use the first blade's pressure-side camberline
    # Each spanwise strip becomes a section
    if blade_surfaces:
        first_blade = blade_surfaces[0]
        # ps is List[List[BladePoint3D]] — outer = spanwise, inner = chordwise
        ps_spans: list[Any] = []
        if isinstance(first_blade, dict):
            ps_spans = first_blade.get("ps", [])
        else:
            ps_spans = first_blade.ps if hasattr(first_blade, "ps") else []

        for span_idx, span_points in enumerate(ps_spans):
            sections.append(
                _format_blade_section(span_points, span_idx)
            )
            sections.append("")

    # Thickness distribution
    sections.append(_default_thickness_distribution())
    sections.append("")

    # Number of blades
    sections.append("NUMBER OF BLADES")
    sections.append(str(blade_count))

    return "\n".join(sections)
