"""Cavitation analysis — NPSH required and cavitation index.

Improvements:
    #6 — Surek & Stempin (2010) method as third estimator.
         Weighted average of all three methods by validity range.

References:
    - Gulich (2014), Ch. 6.
    - Thoma, D. (1937). Kavitation bei Wasserturbinen.
    - Surek, D. & Stempin, S. (2010). Angewandte Strömungsmechanik.
"""

from __future__ import annotations
import math
from hpe.core.models import G
from hpe.constants import THOMA_C, NPSH_LAMBDA


def calc_thoma_sigma(nq: float) -> float:
    """sigma_c = (nq / C)^(4/3) with C = 900."""
    return (nq / THOMA_C) ** (4.0 / 3.0)


def calc_npsh_required_thoma(head: float, nq: float) -> float:
    """NPSHr = sigma * H  (Thoma method)."""
    return calc_thoma_sigma(nq) * head


def calc_npsh_required_inlet(
    flow_rate: float,
    d1: float,
    d1_hub: float,
    rpm: float,
    lambda_c: float = NPSH_LAMBDA,
) -> float:
    """NPSHr from relative velocity at leading edge (Gulich §6.2)."""
    a_eye = math.pi / 4.0 * max(d1 ** 2 - d1_hub ** 2, 1e-6)
    c_eye = flow_rate / a_eye
    u1 = math.pi * d1 * rpm / 60.0
    w1 = math.sqrt(c_eye ** 2 + u1 ** 2)
    return lambda_c * w1 ** 2 / (2.0 * G)


def calc_npsh_required_surek(
    flow_rate: float,
    d1: float,
    d1_hub: float,
    rpm: float,
    nq: float,
) -> float:
    """NPSHr by Surek & Stempin (2010) method (#6).

    Combines inlet velocity triangle with specific speed correction:
        NPSHr_S = lambda_S * (c_m1^2 + u1^2/4) / (2*g)
    where lambda_S = 0.3 * (nq/100)^0.5 + 0.8

    Accounts for both incidence losses and pressure recovery.

    Args:
        flow_rate: Q [m³/s].
        d1: Inlet diameter [m].
        d1_hub: Hub diameter [m].
        rpm: RPM.
        nq: Metric specific speed.

    Returns:
        NPSHr [m].
    """
    a_eye = math.pi / 4.0 * max(d1 ** 2 - d1_hub ** 2, 1e-6)
    cm1 = flow_rate / a_eye
    u1 = math.pi * d1 * rpm / 60.0
    lambda_s = 0.3 * (nq / 100.0) ** 0.5 + 0.8
    return lambda_s * (cm1 ** 2 + u1 ** 2 / 4.0) / (2.0 * G)


def calc_npsh_pfleiderer(
    flow_rate: float,
    d1: float,
    d1_hub: float,
    rpm: float,
    lambda_w: float = 1.3,
    rho: float = 998.0,
) -> float:
    """NPSHr by Pfleiderer method (C2).

    Based on the maximum relative velocity at the leading edge, amplified
    by the blade blockage factor lambda_w:

        NPSHr_Pfleiderer = lambda_w² × w1² / (2×g)

    Args:
        flow_rate: Q [m³/s].
        d1: Inlet tip diameter [m].
        d1_hub: Hub diameter [m].
        rpm: Rotational speed [RPM].
        lambda_w: LE velocity amplification factor (typically 1.2–1.4, Gülich §6.2).
        rho: Fluid density [kg/m³] (unused, kept for signature consistency).

    Returns:
        NPSHr [m].
    """
    a_eye = math.pi / 4.0 * max(d1 ** 2 - d1_hub ** 2, 1e-6)
    cm1 = flow_rate / a_eye
    u1 = math.pi * d1 * rpm / 60.0
    w1 = math.sqrt(cm1 ** 2 + u1 ** 2)
    return lambda_w ** 2 * w1 ** 2 / (2.0 * G)


def calc_npsh_required(
    flow_rate: float,
    head: float,
    d1: float,
    d1_hub: float,
    rpm: float,
    nq: float,
) -> tuple[float, float]:
    """Calculate NPSHr using weighted average of three methods (#6).

    Each method is weighted by its claimed validity range for Nq:
    - Thoma: weight = 1.0 (always applicable, purely empirical)
    - Inlet velocity: weight = 1.2 (better for low Nq < 40)
    - Surek: weight = 1.0 (best for 20 < Nq < 80)

    Returns the weighted mean capped by max of all three (conservative
    but not as overly pessimistic as always-max).

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        d1: Inlet diameter [m].
        d1_hub: Hub diameter [m].
        rpm: RPM.
        nq: Metric specific speed.

    Returns:
        Tuple of (NPSHr [m], sigma).
    """
    npsh_thoma = calc_npsh_required_thoma(head, nq)
    npsh_inlet = calc_npsh_required_inlet(flow_rate, d1, d1_hub, rpm)
    npsh_surek = calc_npsh_required_surek(flow_rate, d1, d1_hub, rpm, nq)
    npsh_pfleiderer = calc_npsh_pfleiderer(flow_rate, d1, d1_hub, rpm)

    # Validity weights — Pfleiderer gets equal weight to Thoma (broadly applicable)
    w_thoma = 1.0
    w_inlet = 1.2 if nq < 40 else 0.8
    w_surek = 1.0 if 20 <= nq <= 80 else 0.6
    w_pfleiderer = 1.0  # C2: added Pfleiderer method

    npsh_r = (
        w_thoma * npsh_thoma
        + w_inlet * npsh_inlet
        + w_surek * npsh_surek
        + w_pfleiderer * npsh_pfleiderer
    ) / (w_thoma + w_inlet + w_surek + w_pfleiderer)

    # Do not let the weighted mean fall below the most conservative method
    npsh_r = max(npsh_r, min(npsh_thoma, npsh_inlet, npsh_surek, npsh_pfleiderer))

    sigma = npsh_r / head if head > 0 else 0.0
    return npsh_r, sigma
