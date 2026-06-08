"""Splitter blade generation for centrifugal impellers.

Splitter blades are shorter blades placed between the main blades
to improve flow guidance without excessive blockage at the inlet.
They start at some fraction of the meridional chord (typically 40-60%
from inlet) and extend to the trailing edge.

The splitter follows the same camber law as the main blade but with:
    - A shorter chord (starts at a specified meridional position)
    - Optional angular offset from the main blade midpitch position
    - Configurable work ratio (fraction of main blade Euler work)

References:
    - Gui, L. et al. (1989). Effect of splitter blades on performance
      of centrifugal pumps.
    - Miyamoto, H. et al. (1992). Flow optimization of centrifugal
      impellers with splitter blades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpe.geometry.models import BladeProfile, RunnerGeometryParams


@dataclass
class SplitterConfig:
    """Configuration for splitter blades.

    Controls the position, size, and work distribution of splitter blades
    relative to the main blades.
    """

    enabled: bool = False
    count: int = 0  # Number of splitters (typically = main blade count)

    # Meridional start position (0=LE, 1=TE)
    # Typical range: 0.4-0.6 (start at 40-60% chord)
    start_fraction: float = 0.50

    # Angular offset from midpitch [degrees]
    # 0 = exactly centered between main blades
    # Positive = shifted toward suction side of preceding blade
    angular_offset: float = 0.0

    # Work ratio: fraction of main blade ΔrVθ carried by splitter
    # Typical: 0.4-0.6 (splitter does 40-60% of main blade work)
    work_ratio: float = 0.50

    # Thickness ratio relative to main blade
    thickness_ratio: float = 0.80


@dataclass
class SplitterResult:
    """Result of splitter blade generation."""

    profiles: list[BladeProfile]  # One per splitter blade
    count: int
    start_fraction: float
    angular_positions: list[float]  # Angular position of each splitter [rad]
    work_ratio: float


def generate_splitter_blades(
    main_profile: BladeProfile,
    params: RunnerGeometryParams,
    config: SplitterConfig,
) -> SplitterResult:
    """Generate splitter blade profiles from main blade and config.

    The splitter camber line is derived from the main blade by:
    1. Trimming the main blade from start_fraction to TE
    2. Offsetting angularly to the midpitch position
    3. Optionally adjusting angles for the desired work ratio

    Args:
        main_profile: Main blade profile (from generate_blade_profile).
        params: Runner geometry parameters.
        config: Splitter blade configuration.

    Returns:
        SplitterResult with all splitter profiles.
    """
    if not config.enabled or config.count <= 0:
        return SplitterResult(
            profiles=[], count=0, start_fraction=config.start_fraction,
            angular_positions=[], work_ratio=config.work_ratio,
        )

    n_main = params.blade_count
    pitch_angle = 2.0 * math.pi / n_main  # Angular pitch between main blades

    # Find the start index in the main blade profile
    n_pts = len(main_profile.camber_points)
    start_idx = int(config.start_fraction * (n_pts - 1))
    start_idx = max(1, min(start_idx, n_pts - 2))

    # Extract the trimmed portion of the main blade
    trimmed_camber = main_profile.camber_points[start_idx:]

    # Compute angular offset to place splitter at midpitch
    midpitch_offset = pitch_angle / 2.0
    user_offset_rad = math.radians(config.angular_offset)
    total_offset = midpitch_offset + user_offset_rad

    # Adjust blade angles for work ratio if != 1.0
    adjusted_camber = _adjust_for_work_ratio(
        trimmed_camber, config.work_ratio,
    )

    # Apply thickness (thinner than main blade)
    splitter_thickness = params.blade_thickness * config.thickness_ratio

    # Generate profiles for each splitter blade
    profiles: list[BladeProfile] = []
    angular_positions: list[float] = []

    for i in range(config.count):
        # Angular position of the i-th splitter
        main_blade_angle = i * pitch_angle
        splitter_angle = main_blade_angle + total_offset
        angular_positions.append(splitter_angle)

        # Offset the camber line
        offset_camber = [
            (r, theta + splitter_angle)
            for r, theta in adjusted_camber
        ]

        # Apply thickness
        ps, ss = _apply_splitter_thickness(offset_camber, splitter_thickness)

        profiles.append(BladeProfile(
            camber_points=offset_camber,
            pressure_side=ps,
            suction_side=ss,
            thickness=splitter_thickness,
        ))

    return SplitterResult(
        profiles=profiles,
        count=config.count,
        start_fraction=config.start_fraction,
        angular_positions=angular_positions,
        work_ratio=config.work_ratio,
    )


def calc_splitter_effect_on_performance(
    main_blade_count: int,
    splitter_count: int,
    work_ratio: float,
    start_fraction: float,
) -> dict[str, float]:
    """Estimate the effect of splitters on pump performance.

    Returns correction factors to apply to the base sizing.

    Args:
        main_blade_count: Number of main blades.
        splitter_count: Number of splitter blades.
        work_ratio: Splitter work ratio.
        start_fraction: Meridional start fraction.

    Returns:
        Dict with correction factors for efficiency, slip, head.
    """
    # Effective blade count for slip factor calculation
    # Splitters act as partial blades for slip correction
    splitter_effectiveness = (1.0 - start_fraction) * work_ratio
    z_effective = main_blade_count + splitter_count * splitter_effectiveness

    # Blockage effect at outlet (more blades = more blockage)
    outlet_blockage_increase = splitter_count * 0.005  # ~0.5% per splitter

    # Efficiency benefit from better flow guidance (reduced secondary flows)
    # Typical improvement: 1-3% for well-designed splitters
    guidance_factor = 1.0 - start_fraction  # More benefit if splitters start early
    eta_improvement = 0.015 * splitter_count / main_blade_count * guidance_factor

    # Slip factor improvement (more blades = less slip)
    # Using modified Wiesner: sigma increases with effective Z
    slip_improvement = 0.02 * (z_effective - main_blade_count) / main_blade_count

    return {
        "z_effective": z_effective,
        "blockage_increase": outlet_blockage_increase,
        "efficiency_improvement": eta_improvement,
        "slip_improvement": slip_improvement,
        "head_correction": 1.0 + slip_improvement - outlet_blockage_increase * 0.5,
    }


def _adjust_for_work_ratio(
    camber_points: list[tuple[float, float]],
    work_ratio: float,
) -> list[tuple[float, float]]:
    """Adjust splitter blade angles for desired work ratio.

    Work ratio < 1 means less blade turning → the theta progression
    is scaled relative to a straight radial line.

    The adjustment modifies the blade wrap: a work_ratio of 0.5 means
    the splitter produces 50% of the angular momentum change per unit
    meridional length compared to the main blade.
    """
    if abs(work_ratio - 1.0) < 1e-6:
        return list(camber_points)

    if len(camber_points) < 2:
        return list(camber_points)

    # Reference: the theta at inlet of the trimmed section
    r_ref, theta_ref = camber_points[0]

    adjusted: list[tuple[float, float]] = []
    for r, theta in camber_points:
        # Scale the theta deviation from the reference
        delta_theta = theta - theta_ref
        adjusted_theta = theta_ref + delta_theta * work_ratio
        adjusted.append((r, adjusted_theta))

    return adjusted


def _apply_splitter_thickness(
    camber_points: list[tuple[float, float]],
    max_thickness: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Apply thickness to splitter blade camber line.

    Uses the same parabolic distribution as main blades but with
    a blunter leading edge (since the splitter LE is in mid-passage).
    """
    n = len(camber_points)
    ps: list[tuple[float, float]] = []
    ss: list[tuple[float, float]] = []

    for i in range(n):
        r, theta = camber_points[i]
        s = i / max(n - 1, 1)

        # Modified distribution: blunter LE for splitter
        # t(s) = max_t * (1 - (2s - 1)^4)  — flatter in the middle
        half_t = max_thickness / 2.0 * (1.0 - (2.0 * s - 1.0) ** 4)
        half_t = max(0.0, half_t)

        if r > 1e-6:
            dtheta = half_t / r
        else:
            dtheta = 0.0

        ps.append((r, theta + dtheta))
        ss.append((r, theta - dtheta))

    return ps, ss


# ── TD1-style sizing helpers ──────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass, field as _field


@_dataclass
class SplitterSizingResult:
    """Result of TD1-style splitter blade sizing (work-split approach)."""

    enabled: bool
    start_fraction: float
    pitch_offset: float
    work_ratio: float
    splitter_blade_count: int          # same as main blade count (interleaved)
    throat_area_reduction: float       # fractional reduction in throat area
    loading_reduction_percent: float   # estimated blade loading reduction at throat
    passage_width_mm: float            # main passage width at throat [mm]
    splitter_start_m: float            # meridional distance where splitter starts [m]
    warnings: list[str] = _field(default_factory=list)


def size_splitter(
    blade_count: int,
    d2: float,
    d1: float,
    b2: float,
    beta2: float,
    start_fraction: float = 0.4,
    pitch_offset: float = 0.5,
    work_ratio: float = 0.5,
    thickness_ratio: float = 0.8,
) -> SplitterSizingResult:
    """Size splitter blades for a centrifugal impeller (TD1-style work-split).

    This function complements :func:`generate_splitter_blades` by providing
    a higher-level sizing result with throat area and loading estimates, as
    used in the ADT TurboDesign1 workflow.

    Args:
        blade_count: Number of main blades
        d2: Outlet diameter [m]
        d1: Inlet diameter [m]
        b2: Outlet width [m]
        beta2: Outlet blade angle [deg]
        start_fraction: Where splitter starts (fraction of meridional chord)
        pitch_offset: Pitchwise position (0.5 = centred between main blades)
        work_ratio: Work fraction handled by splitter passage
        thickness_ratio: Splitter/main blade thickness ratio

    Returns:
        SplitterSizingResult with sizing metrics and warnings
    """
    import math as _math

    warnings: list[str] = []
    beta2_rad = _math.radians(beta2)

    # Meridional chord estimate: half-circumference of mean diameter / blade_count
    d_mean = (d1 + d2) / 2
    meridional_chord = _math.pi * d_mean / blade_count  # rough estimate [m]
    splitter_start_m = start_fraction * meridional_chord

    # Pitch at outlet
    pitch_d2 = _math.pi * d2 / blade_count  # [m]

    # Throat area reduction: splitter splits each passage into two at throat
    throat_area_reduction = 1.0 - start_fraction  # approximate
    loading_reduction_percent = throat_area_reduction * (1.0 - work_ratio) * 100.0

    # Main passage width at throat
    passage_width_mm = (pitch_d2 * _math.sin(beta2_rad)) * 1000  # mm

    # Validation
    if start_fraction < 0.25:
        warnings.append("Splitter start < 25% meridional chord — may increase leading edge blockage")
    if start_fraction > 0.65:
        warnings.append("Splitter start > 65% — limited benefit, consider omitting")
    if work_ratio < 0.35 or work_ratio > 0.65:
        warnings.append(f"Work ratio {work_ratio:.2f} outside typical 0.35–0.65 range")
    if blade_count * 2 > 18:
        warnings.append(f"Total blade count with splitters ({blade_count * 2}) is high — check blockage")

    return SplitterSizingResult(
        enabled=True,
        start_fraction=start_fraction,
        pitch_offset=pitch_offset,
        work_ratio=work_ratio,
        splitter_blade_count=blade_count,
        throat_area_reduction=throat_area_reduction,
        loading_reduction_percent=loading_reduction_percent,
        passage_width_mm=passage_width_mm,
        splitter_start_m=splitter_start_m,
        warnings=warnings,
    )


def splitter_loading_distribution(
    main_loading: list[float],
    start_fraction: float = 0.4,
    work_ratio: float = 0.5,
) -> list[float]:
    """Compute splitter blade rVθ* loading profile from main blade loading.

    The splitter passage carries work_ratio of total work starting from
    start_fraction of meridional chord. Before that point, only the main
    blade does work.

    Args:
        main_loading: rVθ* values at normalized m stations (0 to 1)
        start_fraction: where splitter starts
        work_ratio: fraction of work in splitter passage

    Returns:
        rVθ* profile for splitter blade (same length as main_loading)
    """
    n = len(main_loading)
    result = []
    for i, rv in enumerate(main_loading):
        m_norm = i / max(1, n - 1)
        if m_norm < start_fraction:
            # Before splitter: no loading on splitter blade
            result.append(0.0)
        else:
            # After splitter: scale main blade loading by work_ratio
            result.append(rv * work_ratio)
    return result
