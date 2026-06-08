"""Closed meridional channel CAD solid (#13).

Generates a proper 3D solid whose hub and shroud surfaces follow the
parametric meridional channel profiles (hub_points, shroud_points) from
``generate_meridional_channel()``, rather than using a simplified fixed-
thickness approximation.

The solid is a body of revolution (axisymmetric) obtained by:
1. Tracing the shroud curve (outer wall) from inlet to outlet in the (r,z) plane.
2. Adding a thin wall outward (disk thickness = ``disk_t``).
3. Tracing back along the hub curve (inner wall).
4. Closing the profile at inlet and outlet.
5. Revolving the closed 2D polygon 360° around the Z axis.

The resulting solid represents the "impeller disk + channel walls" — correct
for STEP/IGES export to CFD meshing software.

Coordinate convention (CadQuery XZ plane):
    r → X  (radial, horizontal)
    z → Z  (axial, positive = inlet side)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cadquery as cq

from hpe.geometry.models import MeridionalChannel


def generate_closed_channel_solid(
    channel: MeridionalChannel,
    disk_t: float = 5.0,
    unit: str = "m",
) -> "cq.Workplane":
    """Build a closed meridional channel solid using CadQuery.

    Creates a body of revolution by tracing:
      shroud curve → outlet wall → hub curve (reversed) → inlet wall → close

    The extra ``disk_t`` thickness is added on the *back* (negative-z side)
    of the shroud profile to give the shroud disk structural depth.

    Args:
        channel: MeridionalChannel with hub_points and shroud_points in (r, z).
        disk_t: Wall thickness of shroud/hub disks [same unit as channel
            points, auto-scaled if unit == "m"].
        unit: "m" — automatically converts to mm for CadQuery.

    Returns:
        CadQuery Workplane containing the closed meridional channel solid.
    """
    import cadquery as cq

    scale = 1000.0 if unit == "m" else 1.0  # m → mm
    t = disk_t * scale

    hub_pts = [(r * scale, z * scale) for r, z in channel.hub_points]
    shr_pts = [(r * scale, z * scale) for r, z in channel.shroud_points]

    # Ensure outlet is the last point (smallest z, usually)
    # Convention: points ordered inlet (high-z) → outlet (low-z)

    # Build closed 2D polygon in the (r, z) = (X, Z) plane:
    # [0] Start at shroud inlet (r_s0, z_s0)
    # [1] Shroud curve: inlet → outlet
    # [2] Outlet wall: shroud outlet → hub outlet (same z, different r)
    # [3] Hub curve (reversed): outlet → inlet
    # [4] Inlet wall: hub inlet → back to shroud inlet, via a thin wall
    # NOTE: we add a back-disk offset on the shroud side so the solid
    #       has wall thickness (not zero-area shell).

    shr_r0, shr_z0 = shr_pts[0]   # shroud inlet
    shr_rn, shr_zn = shr_pts[-1]  # shroud outlet
    hub_r0, hub_z0 = hub_pts[0]   # hub inlet
    hub_rn, hub_zn = hub_pts[-1]  # hub outlet

    # Disk back offsets
    shr_z_back = shr_z0 + t   # shroud inlet back face (axially upstream)
    hub_z_back = hub_z0 + t

    # Build polygon as list of (r, z) vertices for a closed wire
    poly: list[tuple[float, float]] = []

    # A: shroud curve (inlet to outlet) — outer wall
    poly.extend(shr_pts)

    # B: outlet wall — shroud outlet down to hub outlet (straight radial)
    # Interpolate a few points for smooth face (just two is fine for revolve)
    poly.append((hub_rn, hub_zn))

    # C: hub curve (outlet to inlet) — inner wall, reversed
    poly.extend(reversed(hub_pts))

    # D: inlet wall (hub inlet back → hub inlet → gap → shroud inlet → shroud inlet back)
    # Add a thin back-disk connector to close the channel solidly
    poly.append((hub_r0, hub_z_back))
    poly.append((shr_r0, shr_z_back))
    poly.append(shr_pts[0])  # back to start

    # Deduplicate consecutive identical points
    clean: list[tuple[float, float]] = [poly[0]]
    for pt in poly[1:]:
        if abs(pt[0] - clean[-1][0]) > 1e-6 or abs(pt[1] - clean[-1][1]) > 1e-6:
            clean.append(pt)

    # Build CadQuery wire in XZ plane, then revolve around Z axis
    wire = (
        cq.Workplane("XZ")
        .moveTo(clean[0][0], clean[0][1])
    )
    for r, z in clean[1:]:
        wire = wire.lineTo(r, z)
    wire = wire.close()

    solid = wire.revolve(
        angleDegrees=360,
        axisStart=(0, 0, 0),
        axisEnd=(0, 0, 1),
    )

    return solid


def generate_inlet_eye_solid(
    channel: MeridionalChannel,
    eye_depth: float = 0.02,
    unit: str = "m",
) -> "cq.Workplane":
    """Build the inlet eye (axial inlet cylinder) as a revolved solid.

    This represents the inlet passage from the suction flange to the
    impeller eye, used when assembling a complete hydraulic model.

    Args:
        channel: Meridional channel.
        eye_depth: Axial depth of the inlet cylinder [m if unit=="m"].
        unit: "m" converts to mm.

    Returns:
        CadQuery Workplane — an annular cylinder at the inlet.
    """
    import cadquery as cq

    scale = 1000.0 if unit == "m" else 1.0
    depth = eye_depth * scale

    shr_r0 = channel.shroud_points[0][0] * scale
    shr_z0 = channel.shroud_points[0][1] * scale
    hub_r0 = channel.hub_points[0][0] * scale
    hub_z0 = channel.hub_points[0][1] * scale

    z_top = max(shr_z0, hub_z0) + depth

    # Annular ring: shroud outer - hub inner, axial extent = depth
    poly = [
        (hub_r0, hub_z0),
        (hub_r0, z_top),
        (shr_r0, z_top),
        (shr_r0, shr_z0),
    ]

    wire = (
        cq.Workplane("XZ")
        .moveTo(poly[0][0], poly[0][1])
    )
    for r, z in poly[1:]:
        wire = wire.lineTo(r, z)
    wire = wire.close()

    return wire.revolve(360, (0, 0, 0), (0, 0, 1))
