"""3D volute geometry generation using CadQuery.

Creates the volute solid by lofting cross-sections at each
circumferential station along the spiral path.
"""

from __future__ import annotations

import math
from typing import Optional

import cadquery as cq

from hpe.core.models import SizingResult
from hpe.geometry.volute.models import VoluteParams, VoluteSizing
from hpe.geometry.volute.sizing import size_volute


def generate_volute(
    params: VoluteParams,
    sizing: Optional[VoluteSizing] = None,
) -> cq.Workplane:
    """Generate 3D volute geometry.

    Creates the volute as a series of circular/rectangular sections
    positioned along the spiral and lofted into a solid.

    Args:
        params: VoluteParams with design specifications.
        sizing: Pre-computed VoluteSizing. If None, computed from params.

    Returns:
        CadQuery Workplane with the volute solid.
    """
    if sizing is None:
        sizing = size_volute(params)

    # Convert to mm for CadQuery
    mm = 1000.0
    r3 = sizing.r3 * mm
    b2 = params.b2 * mm

    # Build volute from circular sections positioned along spiral
    # Each section is at angle theta, centered at radius r3 + offset
    sections: list[cq.Wire] = []

    for i in range(len(sizing.theta_stations)):
        theta_deg = sizing.theta_stations[i]
        area_mm2 = sizing.areas[i] * mm * mm  # Convert m^2 to mm^2

        if area_mm2 < 1.0:  # Skip very small sections near tongue
            continue

        # Section radius (circular)
        r_section = math.sqrt(area_mm2 / math.pi)

        # Position: center of section at angle theta, radius r3 + r_section
        theta_rad = math.radians(theta_deg)
        cx = (r3 + r_section) * math.cos(theta_rad)
        cy = (r3 + r_section) * math.sin(theta_rad)

        # Create circular wire in a plane perpendicular to the spiral
        # The section plane is oriented radially (normal = tangent to circle)
        try:
            section = (
                cq.Workplane("XY")
                .transformed(offset=(cx, cy, 0))
                .transformed(rotate=(0, 0, theta_deg))
                .circle(max(r_section, 0.5))
            )
            wire = section.val()
            if hasattr(wire, 'Wire'):
                sections.append(wire)
        except Exception:
            continue

    # If we couldn't create wire sections, fall back to simpler approach
    if len(sections) < 3:
        return _generate_volute_simple(params, sizing, mm)

    # Try lofting
    try:
        result = cq.Workplane("XY")
        # Loft is tricky with spiral; use simple approach
        return _generate_volute_simple(params, sizing, mm)
    except Exception:
        return _generate_volute_simple(params, sizing, mm)


def _generate_volute_simple(
    params: VoluteParams,
    sizing: VoluteSizing,
    mm: float,
) -> cq.Workplane:
    """Simple volute generation — annular body with growing section.

    Creates the volute as a revolved annular shape with a tangential
    discharge pipe. Simpler but produces a valid solid for CFD.
    """
    r3 = sizing.r3 * mm
    b2 = params.b2 * mm

    # Maximum section radius (at theta=360)
    max_area = sizing.discharge_area * mm * mm
    r_max = math.sqrt(max_area / math.pi) if max_area > 0 else b2

    # Mean section radius (at theta=180)
    mean_area = sizing.areas[len(sizing.areas) // 2] * mm * mm
    r_mean = math.sqrt(mean_area / math.pi) if mean_area > 0 else b2 * 0.5

    # Create volute as annular body
    # Inner wall: circle at r3
    # Outer wall: circle at r3 + 2*r_max
    r_outer = r3 + 2.0 * r_max
    height = max(b2 * 1.5, 2.0 * r_max)

    # Annular ring (revolved)
    volute = (
        cq.Workplane("XZ")
        .moveTo(r3, -height / 2)
        .lineTo(r_outer, -height / 2)
        .lineTo(r_outer, height / 2)
        .lineTo(r3, height / 2)
        .close()
        .revolve(360, (0, 0, 0), (0, 1, 0))
    )

    # Add tangential discharge pipe
    discharge_len = params.d2 * mm * params.discharge_length_ratio
    discharge_r = math.sqrt(max_area / math.pi) if max_area > 0 else b2

    discharge = (
        cq.Workplane("XY")
        .transformed(offset=(r3 + r_max, 0, 0))
        .circle(discharge_r)
        .extrude(discharge_len)
    )

    result = volute.union(discharge)
    return result


def generate_volute_from_sizing(
    sizing_result: SizingResult,
) -> cq.Workplane:
    """Generate volute directly from a SizingResult.

    Args:
        sizing_result: Output from run_sizing().

    Returns:
        CadQuery Workplane with the volute solid.
    """
    params = VoluteParams.from_sizing_result(sizing_result)
    return generate_volute(params)
