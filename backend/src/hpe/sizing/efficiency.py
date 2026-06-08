"""Efficiency estimation via empirical correlations.

Estimates hydraulic, volumetric, mechanical, and overall efficiency
based on operating parameters and specific speed. These correlations
provide starting points for sizing; actual efficiency is determined
by CFD or test bench measurements.

References:
    - Gulich, J.F. (2014). Centrifugal Pumps, 3rd ed. Springer, Ch. 3.
    - Stepanoff, A.J. (1957). Centrifugal and Axial Flow Pumps.
"""

from __future__ import annotations



def estimate_hydraulic_efficiency(flow_rate: float, nq: float) -> float:
    """Estimate hydraulic efficiency using Gulich correlation.

    Based on Gulich (2014) Eq. 3.28 (simplified):
        eta_h = 1 - 0.055 * (Q_ref / Q)^0.25 - 0.002 * (Q_ref / nq)^0.5

    For preliminary sizing we use a simplified version calibrated
    for centrifugal pumps in the range Nq = 15-80, Q = 0.005-1.0 m3/s.

    Args:
        flow_rate: Q [m^3/s].
        nq: Metric specific speed Nq.

    Returns:
        Estimated hydraulic efficiency eta_h (0 < eta_h < 1).
    """
    # Reference flow rate for normalization (Gulich)
    q_ref = 1.0  # m3/s

    # Gulich-based correlation
    eta_h = 1.0 - 0.055 * (q_ref / max(flow_rate, 1e-6)) ** 0.25

    # Correction for specific speed (penalty at very low or very high nq)
    nq_opt = 40.0  # Optimal nq for best efficiency
    nq_penalty = 0.003 * ((nq - nq_opt) / nq_opt) ** 2
    eta_h -= nq_penalty

    return max(0.5, min(0.96, eta_h))


def estimate_volumetric_efficiency(nq: float) -> float:
    """Estimate volumetric efficiency based on specific speed.

    Leakage losses decrease with increasing specific speed because
    the ratio of leakage area to through-flow area decreases.

    Gulich (2014) Eq. 3.34:
        eta_v = 1 / (1 + 0.68 * nq^(-2/3))

    Args:
        nq: Metric specific speed Nq.

    Returns:
        Estimated volumetric efficiency eta_v.
    """
    eta_v = 1.0 / (1.0 + 0.68 * nq ** (-2.0 / 3.0))
    return max(0.80, min(0.99, eta_v))


def estimate_mechanical_efficiency(flow_rate: float, nq: float) -> float:
    """Estimate mechanical efficiency (disk friction + bearing losses).

    Disk friction losses decrease with increasing specific speed.
    Bearing and seal losses are relatively constant.

    Args:
        flow_rate: Q [m^3/s].
        nq: Metric specific speed Nq.

    Returns:
        Estimated mechanical efficiency eta_m.
    """
    # Disk friction component (decreases with nq)
    disk_loss_fraction = 0.02 * (40.0 / max(nq, 10.0)) ** 0.5

    # Bearing/seal losses (roughly constant for a given size)
    bearing_loss_fraction = 0.01

    eta_m = 1.0 - disk_loss_fraction - bearing_loss_fraction
    return max(0.85, min(0.99, eta_m))


def estimate_all_efficiencies(
    flow_rate: float,
    nq: float,
) -> tuple[float, float, float, float]:
    """Estimate all efficiency components.

    Args:
        flow_rate: Q [m^3/s].
        nq: Metric specific speed Nq.

    Returns:
        Tuple of (eta_h, eta_v, eta_m, eta_total).
    """
    eta_h = estimate_hydraulic_efficiency(flow_rate, nq)
    eta_v = estimate_volumetric_efficiency(nq)
    eta_m = estimate_mechanical_efficiency(flow_rate, nq)
    eta_total = eta_h * eta_v * eta_m
    return eta_h, eta_v, eta_m, eta_total
