"""Performance curve generation (H-Q, eta-Q, P-Q, NPSH-Q).

Improvements:
    #3 — H-Q curve now recomputes velocity triangles at each flow rate
         for physically correct off-design behaviour.
    #4 — Unstable zone detection: flags points where dH/dQ > 0.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from hpe.core.models import PerformanceMetrics, SizingResult
from hpe.physics.euler import get_design_flow_rate
from hpe.physics.performance import evaluate_performance
from hpe.sizing.velocity_triangles import (
    calc_inlet_triangle, calc_outlet_triangle, calc_euler_head,
    calc_blockage_factor,
)
from hpe.sizing.efficiency import estimate_all_efficiencies
from hpe.sizing.specific_speed import calc_specific_speed


@dataclass
class PerformanceCurves:
    """Complete set of performance curves for a pump."""
    flow_rates: list[float]
    heads: list[float]
    efficiencies: list[float]
    powers: list[float]
    npsh_required: list[float]
    hydraulic_efficiencies: list[float]
    torques: list[float]
    is_unstable: list[bool] = field(default_factory=list)   # (#4)
    unstable_q_range: tuple[float, float] | None = None     # (#4)
    metrics: list[PerformanceMetrics] = field(default_factory=list)


def _head_from_triangles(
    sizing: SizingResult,
    flow_rate: float,
    eta_h: float,
    rho: float,
) -> float:
    """Compute H at given Q by recalculating velocity triangles (#3).

    Uses Euler equation corrected by hydraulic efficiency:
        H = eta_h * H_euler
    """
    mp = sizing.meridional_profile
    d1 = sizing.impeller_d1
    d2 = sizing.impeller_d2
    b2 = sizing.impeller_b2
    b1 = float(mp.get("b1", b2 * 1.2))
    blade_count = sizing.blade_count
    beta2 = sizing.beta2

    # Re-derive rpm from u2 (stored implicitly via D2 and the design conditions)
    # We need rpm — store it if not in sizing. Fallback: re-derive from u2.
    # u2 = pi*D2*n/60 → n = 60*u2/(pi*D2)
    # u2 at design: u2^2 = 2*g*H_euler_design / (psi*eta_h)
    # Simpler: get rpm from sizing.velocity_triangles
    vt = sizing.velocity_triangles
    u2_design = vt.get("outlet", {}).get("u", None)
    if u2_design and d2 > 0:
        rpm = 60.0 * u2_design / (math.pi * d2)
    else:
        return 0.0  # Cannot compute without rpm

    tau1 = calc_blockage_factor(d1, b1, blade_count, is_inlet=True)
    tau2 = calc_blockage_factor(d2, b2, blade_count, is_inlet=False)

    tri_in = calc_inlet_triangle(d1=d1, b1=b1, flow_rate=flow_rate, rpm=rpm,
                                  blockage_factor=tau1, blade_count=blade_count)
    tri_out = calc_outlet_triangle(d2=d2, b2=b2, flow_rate=flow_rate, rpm=rpm,
                                    beta2=beta2, blockage_factor=tau2, blade_count=blade_count)

    h_euler = calc_euler_head(tri_in, tri_out)
    return max(0.0, eta_h * h_euler)


def generate_curves(
    sizing: SizingResult,
    q_min_ratio: float = 0.1,
    q_max_ratio: float = 1.5,
    n_points: int = 25,
    rho: float = 998.2,
) -> PerformanceCurves:
    """Generate complete performance curves.

    Uses velocity triangle recalculation for H (#3) and
    detects the unstable zone where dH/dQ > 0 (#4).
    """
    q_design = get_design_flow_rate(sizing)
    _, nq = calc_specific_speed(q_design, sizing.impeller_d2, 1.0)  # approximate
    eta_h, _, _, _ = estimate_all_efficiencies(q_design, sizing.specific_speed_nq)

    flow_rates: list[float] = []
    heads: list[float] = []
    efficiencies: list[float] = []
    powers: list[float] = []
    npsh_vals: list[float] = []
    eta_h_vals: list[float] = []
    torques: list[float] = []
    metrics: list[PerformanceMetrics] = []

    for i in range(n_points):
        ratio = q_min_ratio + (q_max_ratio - q_min_ratio) * i / (n_points - 1)
        q = q_design * ratio

        # Use velocity-triangle-based H (#3)
        h_vt = _head_from_triangles(sizing, q, eta_h, rho)

        # Fall back to physics model for efficiency, power, NPSH
        perf = evaluate_performance(sizing, q, rho)

        # Override head with physics-based VT result when valid
        head = h_vt if h_vt > 0 else perf.head

        flow_rates.append(q)
        heads.append(head)
        efficiencies.append(perf.total_efficiency)
        powers.append(perf.power)
        npsh_vals.append(perf.npsh_required)
        eta_h_vals.append(perf.hydraulic_efficiency)
        torques.append(perf.torque)
        metrics.append(perf)

    # Detect unstable zone (#4): dH/dQ > 0 means positive slope → unstable
    is_unstable = [False] * n_points
    unstable_q_range: tuple[float, float] | None = None
    unstable_start = None

    for i in range(1, n_points):
        dq = flow_rates[i] - flow_rates[i - 1]
        dh = heads[i] - heads[i - 1]
        if dq > 0 and dh > 0:  # positive slope = unstable
            is_unstable[i] = True
            if unstable_start is None:
                unstable_start = flow_rates[i - 1]
        else:
            if unstable_start is not None:
                unstable_q_range = (unstable_start, flow_rates[i - 1])
                unstable_start = None

    # Mark PerformanceMetrics with instability flag
    for i, m in enumerate(metrics):
        m.is_unstable = is_unstable[i]

    return PerformanceCurves(
        flow_rates=flow_rates,
        heads=heads,
        efficiencies=efficiencies,
        powers=powers,
        npsh_required=npsh_vals,
        hydraulic_efficiencies=eta_h_vals,
        torques=torques,
        is_unstable=is_unstable,
        unstable_q_range=unstable_q_range,
        metrics=metrics,
    )


def generate_hq_curve(
    sizing: SizingResult,
    q_min_ratio: float = 0.1,
    q_max_ratio: float = 1.5,
    n_points: int = 25,
) -> list[tuple[float, float]]:
    curves = generate_curves(sizing, q_min_ratio, q_max_ratio, n_points)
    return list(zip(curves.flow_rates, curves.heads))


def generate_efficiency_curve(
    sizing: SizingResult,
    q_min_ratio: float = 0.1,
    q_max_ratio: float = 1.5,
    n_points: int = 25,
) -> list[tuple[float, float]]:
    curves = generate_curves(sizing, q_min_ratio, q_max_ratio, n_points)
    return list(zip(curves.flow_rates, curves.efficiencies))
