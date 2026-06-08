"""Distributor (guide vanes) parametric geometry generation.

Creates a ring of stationary guide vanes between the impeller outlet
and the volute inlet. Guide vanes control the flow angle entering
the volute, improving efficiency and reducing pressure pulsations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cadquery as cq

from hpe.core.models import SizingResult


@dataclass
class DistributorParams:
    """Parameters for guide vane geometry."""

    d2: float  # Impeller outlet diameter [m]
    b2: float  # Impeller outlet width [m]
    n_vanes: int = 11  # Number of vanes (prime to avoid resonance with blades)
    vane_angle: float = 15.0  # Vane angle from radial [deg]
    radial_gap_ratio: float = 0.03  # Gap between impeller and vanes (fraction of D2)
    vane_length_ratio: float = 0.15  # Vane radial length as fraction of D2
    vane_thickness: float = 0.003  # Vane thickness [m]

    @property
    def r_inner(self) -> float:
        """Inner radius of guide vane ring."""
        return self.d2 / 2.0 * (1.0 + self.radial_gap_ratio)

    @property
    def r_outer(self) -> float:
        """Outer radius of guide vane ring."""
        return self.r_inner + self.d2 * self.vane_length_ratio

    @classmethod
    def from_sizing_result(cls, sizing_result: SizingResult) -> DistributorParams:
        """Create from SizingResult."""
        return cls(
            d2=sizing_result.impeller_d2,
            b2=sizing_result.impeller_b2,
        )


def generate_distributor(
    params: DistributorParams,
) -> cq.Workplane:
    """Generate a ring of guide vanes.

    Each vane is a thin curved plate spanning from r_inner to r_outer,
    with height b2 and the specified vane angle.

    Args:
        params: DistributorParams.

    Returns:
        CadQuery Workplane with the guide vane ring.
    """
    mm = 1000.0  # CadQuery uses mm
    r_in = params.r_inner * mm
    r_out = params.r_outer * mm
    height = params.b2 * mm * 1.2  # Slightly taller than impeller
    thickness = params.vane_thickness * mm
    angle_rad = math.radians(params.vane_angle)

    # Create one vane as a thin swept solid
    # Vane follows a line from (r_in, 0) to (r_out, r_out*tan(angle))
    dx = r_out - r_in
    dy = dx * math.tan(angle_rad)

    vane = (
        cq.Workplane("XY")
        .transformed(offset=(r_in, 0, -height / 2))
        .rect(dx, thickness)
        .extrude(height)
        .rotate((0, 0, 0), (0, 0, 1), math.degrees(math.atan2(dy, dx)) / 2)
    )

    # Pattern all vanes
    result = vane
    angle_step = 360.0 / params.n_vanes
    for i in range(1, params.n_vanes):
        rotated = vane.rotate((0, 0, 0), (0, 0, 1), angle_step * i)
        result = result.union(rotated)

    return result


def generate_distributor_from_sizing(
    sizing_result: SizingResult,
) -> cq.Workplane:
    """Generate distributor directly from SizingResult."""
    params = DistributorParams.from_sizing_result(sizing_result)
    return generate_distributor(params)
