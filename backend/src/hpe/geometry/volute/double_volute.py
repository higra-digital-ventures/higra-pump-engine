"""Double volute and twin-entry volute design.

Double volute: A single volute with an internal splitter (dividing rib)
that creates two 180° passages, reducing radial forces at off-design.

Twin-entry volute: Two separate inlets feeding into a common discharge,
used in turbines to handle pulsating inlet conditions.

Both types share the same area distribution methodology as single
volutes but with modified angular ranges and merged discharge sections.

References:
    - Gulich, J.F. (2014). Centrifugal Pumps, 3rd ed., Ch. 7.3.
    - Ayder, E. & Van den Braembussche, R. (1994). Numerical analysis
      of double volute centrifugal pumps.
    - Stepanoff, A.J. (1957). Centrifugal and Axial Flow Pumps, Ch. 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hpe.geometry.volute.models import CrossSectionType, VoluteParams, VoluteSizing


class VoluteType(str, Enum):
    """Volute configuration type."""

    SINGLE = "single"
    DOUBLE = "double"  # Internal splitter at 180°
    TWIN_ENTRY = "twin_entry"  # Two separate inlets (turbine)


@dataclass
class DoubleVoluteParams:
    """Parameters for double or twin-entry volute design."""

    # Base volute parameters
    base: VoluteParams

    # Type
    volute_type: VoluteType = VoluteType.DOUBLE

    # Splitter (double volute)
    splitter_angle: float = 180.0  # Angular position of splitter [deg]
    splitter_thickness: float = 0.005  # Splitter wall thickness [m]
    splitter_gap_ratio: float = 0.02  # Gap at tongue side as fraction of D2

    # Twin entry
    entry_flow_split: float = 0.50  # Fraction of total flow in entry A

    # Discharge merge
    merge_length_ratio: float = 1.0  # Length of merge section / D2
    merge_area_ratio: float = 1.1  # Merge outlet area / sum of inlet areas (diffusion)

    # Draft angle for casting [deg]
    draft_angle: float = 3.0


@dataclass
class DoubleVoluteResult:
    """Result of double/twin-entry volute sizing."""

    # Passage A (0° → splitter_angle)
    passage_a: VoluteSizing

    # Passage B (splitter_angle → 360°)
    passage_b: VoluteSizing

    # Radial force reduction
    radial_force_ratio: float  # F_double / F_single (< 1 = improvement)

    # Merge section
    merge_inlet_area_a: float  # [m²]
    merge_inlet_area_b: float  # [m²]
    merge_outlet_area: float  # [m²]

    # Splitter geometry
    splitter_r_inner: float  # Inner radius of splitter [m]
    splitter_r_outer: float  # Outer radius of splitter [m]

    # Total discharge
    total_discharge_area: float  # [m²]


def size_double_volute(
    params: DoubleVoluteParams,
) -> DoubleVoluteResult:
    """Size a double volute with internal splitter.

    The double volute is divided into two 180° passages by a splitter.
    Each passage collects half the flow and delivers it to a common
    discharge pipe. This nearly eliminates the radial force imbalance
    that exists in single volutes at off-design.

    Args:
        params: Double volute design parameters.

    Returns:
        DoubleVoluteResult with both passages and merge section.
    """
    base = params.base
    splitter_deg = params.splitter_angle

    if params.volute_type == VoluteType.TWIN_ENTRY:
        return _size_twin_entry(params)

    # Passage A: 0° to splitter_angle
    passage_a = _size_passage(
        base, theta_start=0.0, theta_end=splitter_deg,
        flow_fraction=0.5,
    )

    # Passage B: splitter_angle to 360°
    passage_b = _size_passage(
        base, theta_start=splitter_deg, theta_end=360.0,
        flow_fraction=0.5,
    )

    # Radial force reduction
    # Single volute: F ∝ ΔP * projected_area (unbalanced over 360°)
    # Double volute: two 180° passages create nearly balanced forces
    # Typical reduction: 70-85%
    radial_force_ratio = _estimate_radial_force_ratio(splitter_deg)

    # Merge section
    area_a = passage_a.discharge_area
    area_b = passage_b.discharge_area
    merge_out = (area_a + area_b) * params.merge_area_ratio

    # Splitter extent: from r3 to max outer radius of passages
    r3 = base.r3
    max_r_a = max(passage_a.radii) if passage_a.radii else r3
    max_r_b = max(passage_b.radii) if passage_b.radii else r3
    splitter_r_outer = max(max_r_a, max_r_b) * 0.95

    return DoubleVoluteResult(
        passage_a=passage_a,
        passage_b=passage_b,
        radial_force_ratio=radial_force_ratio,
        merge_inlet_area_a=area_a,
        merge_inlet_area_b=area_b,
        merge_outlet_area=merge_out,
        splitter_r_inner=r3 + params.splitter_thickness,
        splitter_r_outer=splitter_r_outer,
        total_discharge_area=merge_out,
    )


def calc_radial_force_single(
    params: VoluteParams,
    q_actual: float,
    rho: float = 998.2,
) -> float:
    """Calculate radial force on impeller for a single volute.

    At off-design, non-uniform pressure distribution around the
    impeller creates a net radial force.

    F_r = k * ρ * g * H * D2 * b2 * (1 - Q/Q_design)

    Args:
        params: Volute parameters.
        q_actual: Actual flow rate [m³/s].
        rho: Fluid density [kg/m³].

    Returns:
        Radial force magnitude [N].
    """
    from hpe.core.models import G

    k = 0.36  # Empirical coefficient (Stepanoff)
    q_design = params.flow_rate
    flow_ratio = q_actual / q_design if q_design > 0 else 0

    # Pressure from cu2
    h_approx = params.cu2**2 / (2.0 * G)

    f_r = k * rho * G * h_approx * params.d2 * params.b2 * abs(1.0 - flow_ratio)
    return f_r


def calc_radial_force_double(
    params: DoubleVoluteParams,
    q_actual: float,
    rho: float = 998.2,
) -> float:
    """Calculate radial force for double volute.

    The double volute reduces radial force by creating two opposing
    pressure distributions that partially cancel.

    Args:
        params: Double volute parameters.
        q_actual: Actual flow rate [m³/s].
        rho: Fluid density [kg/m³].

    Returns:
        Radial force magnitude [N].
    """
    f_single = calc_radial_force_single(params.base, q_actual, rho)
    ratio = _estimate_radial_force_ratio(params.splitter_angle)
    return f_single * ratio


def _size_passage(
    base: VoluteParams,
    theta_start: float,
    theta_end: float,
    flow_fraction: float,
) -> VoluteSizing:
    """Size a single passage of the double volute.

    Each passage collects flow_fraction * Q over its angular range.
    """
    r3 = base.r3
    Q = base.flow_rate * flow_fraction
    cu2 = base.cu2
    r2 = base.d2 / 2.0
    K = r2 * cu2

    theta_range = theta_end - theta_start
    n_stations = max(2, int(base.n_stations * theta_range / 360.0))

    theta_stations: list[float] = []
    areas: list[float] = []
    radii: list[float] = []
    widths: list[float] = []

    for i in range(n_stations + 1):
        theta_deg = theta_start + i * theta_range / n_stations
        theta_local = math.radians(theta_deg - theta_start)

        Q_theta = Q * theta_local / (math.radians(theta_range))

        V_mean = K / r3 if r3 > 0 else 1.0
        area = Q_theta / V_mean if V_mean > 0 else 0.0

        tongue_min_area = 0.001 * base.b2 * base.d2
        local_deg = theta_deg - theta_start
        area = max(area, tongue_min_area * min(1.0, local_deg / 30.0))

        if base.cross_section == CrossSectionType.CIRCULAR:
            r_section = math.sqrt(area / math.pi) if area > 0 else 0
            width = 2.0 * r_section
        else:
            width = area / base.b2 if base.b2 > 0 else 0
            r_section = width / 2.0

        r_outer = r3 + 2.0 * r_section

        theta_stations.append(theta_deg)
        areas.append(area)
        radii.append(r_outer)
        widths.append(width)

    discharge_area = areas[-1] if areas else 0.0

    return VoluteSizing(
        theta_stations=theta_stations,
        areas=areas,
        radii=radii,
        widths=widths,
        r3=r3,
        discharge_area=discharge_area,
    )


def _size_twin_entry(
    params: DoubleVoluteParams,
) -> DoubleVoluteResult:
    """Size a twin-entry volute (for turbines).

    Twin entry has two separate inlets, each collecting a fraction
    of the total flow. The flow split can be uneven.
    """
    base = params.base
    split = params.entry_flow_split

    # Entry A: 0-360° with flow_fraction = split
    passage_a = _size_passage(base, 0.0, 360.0, split)

    # Entry B: 0-360° with flow_fraction = (1-split)
    passage_b = _size_passage(base, 0.0, 360.0, 1.0 - split)

    # Twin entry has moderate radial force reduction (not as good as double)
    radial_force_ratio = 0.50 if abs(split - 0.5) < 0.1 else 0.70

    area_a = passage_a.discharge_area
    area_b = passage_b.discharge_area
    merge_out = (area_a + area_b) * params.merge_area_ratio

    r3 = base.r3

    return DoubleVoluteResult(
        passage_a=passage_a,
        passage_b=passage_b,
        radial_force_ratio=radial_force_ratio,
        merge_inlet_area_a=area_a,
        merge_inlet_area_b=area_b,
        merge_outlet_area=merge_out,
        splitter_r_inner=r3,
        splitter_r_outer=r3 * 1.5,  # Approximate
        total_discharge_area=merge_out,
    )


def _estimate_radial_force_ratio(splitter_angle: float) -> float:
    """Estimate radial force reduction ratio for double volute.

    Perfect balance at 180° splitter. Deviation from 180° reduces
    the effectiveness.

    Args:
        splitter_angle: Splitter position [deg].

    Returns:
        Force ratio (double/single), < 1 = improvement.
    """
    # At 180°: ~75% reduction (ratio = 0.25)
    # At 120° or 240°: ~50% reduction (ratio = 0.50)
    # At 90° or 270°: ~30% reduction (ratio = 0.70)
    deviation = abs(splitter_angle - 180.0) / 180.0  # 0 at 180°, 1 at 0/360°
    ratio = 0.20 + 0.60 * deviation**1.5
    return min(1.0, max(0.15, ratio))
