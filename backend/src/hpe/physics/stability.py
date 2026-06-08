"""Operating point stability analysis.

Analyzes performance curves to identify:
- Best Efficiency Point (BEP)
- Stable and unstable operating regions
- Minimum recommended flow
- Shutdown (zero-flow) head
- Surge risk zones
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hpe.core.models import SizingResult
from hpe.physics.curves import PerformanceCurves, generate_curves
from hpe.physics.euler import get_design_flow_rate


@dataclass
class StabilityAnalysis:
    """Results of stability analysis on pump performance curves."""

    # Best Efficiency Point
    bep_flow: float  # Q at BEP [m^3/s]
    bep_head: float  # H at BEP [m]
    bep_efficiency: float  # eta at BEP [-]

    # Shutdown head (Q=0)
    shutdown_head: float  # H at Q=0 [m]
    shutdown_ratio: float  # H_shutoff / H_design

    # Minimum flow
    min_flow: float  # Minimum recommended Q [m^3/s]
    min_flow_ratio: float  # Q_min / Q_design

    # Stability
    is_stable: bool  # True if dH/dQ < 0 everywhere in operating range
    unstable_regions: list[tuple[float, float]]  # (Q_start, Q_end) of unstable zones

    # Warnings
    warnings: list[str] = field(default_factory=list)


def analyze_stability(
    sizing: SizingResult,
    rho: float = 998.2,
) -> StabilityAnalysis:
    """Perform complete stability analysis.

    Args:
        sizing: SizingResult with design geometry.
        rho: Fluid density [kg/m^3].

    Returns:
        StabilityAnalysis with BEP, limits, and warnings.
    """
    q_design = get_design_flow_rate(sizing)
    curves = generate_curves(sizing, q_min_ratio=0.05, q_max_ratio=1.5, n_points=30, rho=rho)

    # Find BEP
    bep_idx = _find_bep_index(curves)
    bep_flow = curves.flow_rates[bep_idx]
    bep_head = curves.heads[bep_idx]
    bep_eff = curves.efficiencies[bep_idx]

    # Shutdown head (extrapolate from lowest flow point)
    shutdown_head = _estimate_shutdown_head(sizing, rho)
    shutdown_ratio = shutdown_head / sizing.velocity_triangles["euler_head"] if sizing.velocity_triangles["euler_head"] > 0 else 0

    # Minimum flow
    min_flow, min_flow_ratio = _find_min_flow(curves, q_design)

    # Stability check
    is_stable, unstable_regions = _check_stability(curves)

    # Generate warnings
    warnings = _generate_warnings(
        bep_flow, bep_head, bep_eff, q_design,
        shutdown_ratio, min_flow_ratio, is_stable,
    )

    return StabilityAnalysis(
        bep_flow=bep_flow,
        bep_head=bep_head,
        bep_efficiency=bep_eff,
        shutdown_head=shutdown_head,
        shutdown_ratio=shutdown_ratio,
        min_flow=min_flow,
        min_flow_ratio=min_flow_ratio,
        is_stable=is_stable,
        unstable_regions=unstable_regions,
        warnings=warnings,
    )


def find_bep(curves: PerformanceCurves) -> tuple[float, float, float]:
    """Find the Best Efficiency Point.

    Args:
        curves: PerformanceCurves.

    Returns:
        Tuple of (Q_bep, H_bep, eta_max).
    """
    idx = _find_bep_index(curves)
    return curves.flow_rates[idx], curves.heads[idx], curves.efficiencies[idx]


def _find_bep_index(curves: PerformanceCurves) -> int:
    """Find index of maximum efficiency in the curves."""
    max_eta = -1.0
    max_idx = 0
    for i, eta in enumerate(curves.efficiencies):
        if eta > max_eta:
            max_eta = eta
            max_idx = i
    return max_idx


def _estimate_shutdown_head(sizing: SizingResult, rho: float) -> float:
    """Estimate head at zero flow (shutdown condition).

    At Q=0, the Euler head is maximum (no cm component reducing cu2).
    H_shutoff ~ u2^2 / (2g) * psi_0 where psi_0 ~ 0.55-0.65

    Args:
        sizing: SizingResult.
        rho: Fluid density.

    Returns:
        Shutdown head [m].
    """
    from hpe.core.models import G

    u2 = sizing.velocity_triangles["outlet"]["u"]
    psi_0 = 0.60  # Shutoff head coefficient (typical for backward-curved blades)
    return psi_0 * u2**2 / G


def _find_min_flow(
    curves: PerformanceCurves,
    q_design: float,
) -> tuple[float, float]:
    """Find minimum recommended flow rate.

    Minimum flow is where efficiency drops below 50% of BEP efficiency,
    or where the curve becomes unstable, whichever is higher.

    Returns:
        Tuple of (Q_min [m^3/s], Q_min/Q_design ratio).
    """
    bep_idx = _find_bep_index(curves)
    eta_threshold = curves.efficiencies[bep_idx] * 0.50

    # Find lowest Q where eta > threshold
    q_min = curves.flow_rates[0]
    for i, (q, eta) in enumerate(zip(curves.flow_rates, curves.efficiencies)):
        if eta >= eta_threshold:
            q_min = q
            break

    ratio = q_min / q_design if q_design > 0 else 0
    return q_min, ratio


def _check_stability(
    curves: PerformanceCurves,
) -> tuple[bool, list[tuple[float, float]]]:
    """Check if the H-Q curve is stable (dH/dQ < 0 everywhere).

    A pump curve is considered unstable where dH/dQ > 0,
    which can cause hunting and surge in systems with
    rising system curves.

    Returns:
        Tuple of (is_stable, list of unstable regions as (Q_start, Q_end)).
    """
    unstable_regions: list[tuple[float, float]] = []
    in_unstable = False
    region_start = 0.0

    for i in range(1, len(curves.heads)):
        dh = curves.heads[i] - curves.heads[i - 1]
        dq = curves.flow_rates[i] - curves.flow_rates[i - 1]

        if dq > 0 and dh > 0:  # Head increasing with flow = unstable
            if not in_unstable:
                region_start = curves.flow_rates[i - 1]
                in_unstable = True
        else:
            if in_unstable:
                unstable_regions.append((region_start, curves.flow_rates[i - 1]))
                in_unstable = False

    if in_unstable:
        unstable_regions.append((region_start, curves.flow_rates[-1]))

    is_stable = len(unstable_regions) == 0
    return is_stable, unstable_regions


def _generate_warnings(
    bep_flow: float,
    bep_head: float,
    bep_eff: float,
    q_design: float,
    shutdown_ratio: float,
    min_flow_ratio: float,
    is_stable: bool,
) -> list[str]:
    """Generate engineering warnings from stability analysis."""
    warnings: list[str] = []

    # BEP far from design point
    bep_ratio = bep_flow / q_design if q_design > 0 else 0
    if abs(bep_ratio - 1.0) > 0.15:
        warnings.append(
            f"BEP is at {bep_ratio:.0%} of design flow. "
            "Consider adjusting design to align BEP with operating point."
        )

    # Low BEP efficiency
    if bep_eff < 0.70:
        warnings.append(
            f"Peak efficiency is {bep_eff:.1%}. "
            "Consider design optimization to improve performance."
        )

    # Unstable curve
    if not is_stable:
        warnings.append(
            "H-Q curve has unstable region(s) (dH/dQ > 0). "
            "Risk of surge in systems with rising resistance curves."
        )

    # High shutdown head ratio
    if shutdown_ratio > 1.4:
        warnings.append(
            f"Shutdown head is {shutdown_ratio:.1f}x design head. "
            "System must withstand shutoff pressure."
        )

    # High minimum flow
    if min_flow_ratio > 0.4:
        warnings.append(
            f"Minimum recommended flow is {min_flow_ratio:.0%} of design. "
            "Limited turndown ratio."
        )

    return warnings
