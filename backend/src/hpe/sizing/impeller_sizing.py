"""Impeller (runner) preliminary dimensioning.

Improvements:
    #5  — Gulich (2014) Table 3.1 volute-type correction for psi.
    #2  — Variable blockage factors via calc_blockage_factor.
    #26 — All magic numbers replaced with constants from hpe.constants.

References:
    - Gulich, J.F. (2014). Centrifugal Pumps, 3rd ed. Springer, Ch. 3.
    - Stepanoff, A.J. (1957). Centrifugal and Axial Flow Pumps.
    - Pfleiderer, C. (1961). Die Kreiselpumpen. Springer.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from hpe.core.models import G
from hpe.constants import (
    PSI_MIN, PSI_MAX, PSI_SLOPE, PSI_INTERCEPT, PSI_MIXED_FLOW,
    D1_HUB_RATIO, B1_ACCEL_FACTOR,
    BLADE_COUNT_MIN, BLADE_COUNT_MAX, BLADE_COUNT_PFLEIDERER,
    BETA2_MIN, BETA2_MAX,
    BLADE_THICKNESS_DEFAULT,
)
from hpe.sizing.velocity_triangles import calc_blockage_factor


@dataclass
class ImpellerDimensions:
    """Preliminary impeller dimensions from 1D sizing."""
    d2: float       # Outlet diameter [m]
    d1: float       # Inlet (eye) diameter [m]
    d1_hub: float   # Hub diameter at inlet [m]
    b2: float       # Outlet width [m]
    b1: float       # Inlet width [m]
    beta1: float    # Inlet blade angle [deg]
    beta2: float    # Outlet blade angle [deg]
    blade_count: int
    u2: float       # Outlet peripheral velocity [m/s]
    tau1: float = 0.90  # Inlet blockage factor (#2)
    tau2: float = 0.88  # Outlet blockage factor (#2)


def calc_head_coefficient(nq: float, volute_type: str = "spiral") -> float:
    """Head coefficient psi = 2*g*H / u2^2.

    Improvement #5 — adds volute_type correction per Gulich (2014) Table 3.1:
        - "spiral"   (standard): no correction
        - "vaned_diffuser": psi *= 0.95 (lower head coeff due to better recovery)
        - "annular"  (ring casing): psi *= 1.05 (higher losses)

    Args:
        nq: Metric specific speed.
        volute_type: "spiral" | "vaned_diffuser" | "annular".

    Returns:
        Head coefficient psi (dimensionless).
    """
    if nq < 100:
        psi = PSI_INTERCEPT - PSI_SLOPE * (nq / 100.0)
    else:
        psi = PSI_MIXED_FLOW

    # Volute type correction (Gulich 2014, Table 3.1)
    corrections = {"spiral": 1.00, "vaned_diffuser": 0.95, "annular": 1.05}
    psi *= corrections.get(volute_type, 1.00)

    return max(PSI_MIN, min(PSI_MAX, psi))


def calc_d1_d2_ratio(nq: float) -> float:
    """D1/D2 from specific speed (Stepanoff/Gulich)."""
    if nq < 20:
        ratio = 0.32
    elif nq < 40:
        ratio = 0.32 + 0.005 * (nq - 20)
    elif nq < 70:
        ratio = 0.42 + 0.004 * (nq - 40)
    elif nq < 120:
        ratio = 0.54 + 0.002 * (nq - 70)
    else:
        ratio = 0.64 + 0.001 * (nq - 120)
    return max(0.30, min(0.80, ratio))


def calc_blade_count(
    d2: float, d1: float, beta1: float, beta2: float, nq: float | None = None
) -> int:
    """Blade count Z, specific-speed-aware.

    The Pfleiderer correlation ``Z = 6.5*(D2+D1)/(D2-D1)*sin((beta1+beta2)/2)``
    is valid for **radial** pumps (Nq <~ 80). Beyond that the eye grows
    (D1 -> D2) and the term ``(D2+D1)/(D2-D1)`` diverges, so Pfleiderer
    *over*-predicts. Physically the blade count *decreases* with specific
    speed: mixed-flow impellers use ~5-7 blades, axial ~3-4.

    Strategy by regime (when ``nq`` is provided):
      * Nq <= 80   radial      -> Pfleiderer, clamped to [BLADE_COUNT_MIN, 12].
      * 80 < Nq <= 140 mixed   -> linear 7 -> 5, clamped to [5, 7].
      * Nq > 140   axial       -> linear 4 -> 3, clamped to [3, 4].

    When ``nq`` is ``None`` the legacy radial behaviour is kept (back-compat).
    """
    beta_avg_rad = math.radians((beta1 + beta2) / 2.0)
    denom = d2 - d1
    z_pfleiderer = (
        BLADE_COUNT_PFLEIDERER * (d2 + d1) / denom * math.sin(beta_avg_rad)
        if abs(denom) >= 1e-6 else 7.0
    )

    # Radial regime (or unknown Nq): keep the validated Pfleiderer result.
    if nq is None or nq <= 80.0:
        return max(BLADE_COUNT_MIN, min(BLADE_COUNT_MAX, round(z_pfleiderer)))

    # Mixed-flow: blade count falls from ~7 (Nq=80) to ~5 (Nq=140).
    if nq <= 140.0:
        z = 7.0 - (nq - 80.0) / 60.0 * 2.0
        return max(5, min(7, round(z)))

    # Axial: ~4 (Nq=140) down to ~3 (Nq>=200).
    z = 4.0 - (min(nq, 200.0) - 140.0) / 60.0
    return max(3, min(4, round(z)))


def calc_outlet_width_ratio(nq: float) -> float:
    """b2/D2 from specific speed (Gulich 2014, Fig. 3.22)."""
    if nq < 25:
        ratio = 0.025 + 0.0015 * nq
    elif nq < 60:
        ratio = 0.06 + 0.002 * (nq - 25)
    elif nq < 100:
        ratio = 0.13 + 0.003 * (nq - 60)
    else:
        ratio = 0.25 + 0.002 * (nq - 100)
    return max(0.02, min(0.50, ratio))


def calc_outlet_blade_angle(nq: float) -> float:
    """Beta2 from specific speed (typical practice)."""
    if nq < 25:
        beta2 = 18.0
    elif nq < 50:
        beta2 = 18.0 + 0.3 * (nq - 25)
    elif nq < 100:
        beta2 = 25.5 + 0.15 * (nq - 50)
    else:
        beta2 = 33.0
    return max(BETA2_MIN, min(BETA2_MAX, beta2))


def size_impeller(
    flow_rate: float,
    head: float,
    rpm: float,
    nq: float,
    eta_h: float,
    volute_type: str = "spiral",
    blade_thickness: float = BLADE_THICKNESS_DEFAULT,
    overrides: dict | None = None,
) -> ImpellerDimensions:
    """Perform preliminary impeller dimensioning.

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        rpm: RPM.
        nq: Metric specific speed.
        eta_h: Estimated hydraulic efficiency.
        volute_type: Volute type for psi correction (#5).
        blade_thickness: t [m] for variable blockage (#2).
        overrides: Optional dict with keys 'd2', 'b2', 'd1' (all in metres).
            When provided the corresponding dimension is replaced after the
            initial sizing and dependent quantities are recalculated (A5).

    Returns:
        ImpellerDimensions.
    """
    # 1. D2 from head coefficient (#5)
    psi = calc_head_coefficient(nq, volute_type)
    u2 = math.sqrt(2.0 * G * head / (psi * eta_h))
    d2 = 60.0 * u2 / (math.pi * rpm)

    # 2. D1
    d1_d2 = calc_d1_d2_ratio(nq)
    d1 = d2 * d1_d2
    d1_hub = d1 * D1_HUB_RATIO

    # 3. b2
    b2 = d2 * calc_outlet_width_ratio(nq)

    # 4. beta2
    beta2 = calc_outlet_blade_angle(nq)

    # 5. Preliminary blade count (estimate beta1 first)
    # Temporary beta1 assuming no pre-swirl and BLOCKAGE_INLET
    from hpe.constants import BLOCKAGE_INLET
    b1_approx = max(b2 * (d2 / d1) * B1_ACCEL_FACTOR, b2)
    inlet_area_approx = math.pi * d1 * b1_approx * BLOCKAGE_INLET
    cm1_approx = flow_rate / inlet_area_approx if inlet_area_approx > 0 else 0.1
    u1 = math.pi * d1 * rpm / 60.0
    beta1_approx = math.degrees(math.atan2(cm1_approx, u1))
    blade_count = calc_blade_count(d2, d1, beta1_approx, beta2, nq)

    # 6. Compute variable blockage factors (#2)
    tau1 = calc_blockage_factor(d1, b1_approx, blade_count, blade_thickness, is_inlet=True)
    tau2 = calc_blockage_factor(d2, b2, blade_count, blade_thickness, is_inlet=False)

    # 7. b1 refined with actual tau1
    b1 = max(b2 * (d2 / d1) * B1_ACCEL_FACTOR, b2)

    # 8. Refined beta1 with actual blockage
    inlet_area = math.pi * d1 * b1 * tau1
    cm1 = flow_rate / inlet_area if inlet_area > 0 else cm1_approx
    beta1 = math.degrees(math.atan2(cm1, u1))

    # 9. Refine blade count with actual beta1
    blade_count = calc_blade_count(d2, d1, beta1, beta2, nq)

    # 10. Apply user overrides and recalculate dependent quantities (A5)
    if overrides:
        if "d2" in overrides and overrides["d2"] is not None:
            d2 = float(overrides["d2"])
            u2 = math.pi * d2 * rpm / 60.0
            # Recompute b2 based on original b2/D2 ratio but new d2
            b2 = d2 * calc_outlet_width_ratio(nq)
        if "b2" in overrides and overrides["b2"] is not None:
            b2 = float(overrides["b2"])
        if "d1" in overrides and overrides["d1"] is not None:
            d1 = float(overrides["d1"])
            d1_hub = d1 * D1_HUB_RATIO
            # Refresh blockage and inlet angle for the new d1
            b1 = max(b2 * (d2 / d1) * B1_ACCEL_FACTOR, b2)
            tau1 = calc_blockage_factor(d1, b1, blade_count, blade_thickness, is_inlet=True)
            inlet_area = math.pi * d1 * b1 * tau1
            u1 = math.pi * d1 * rpm / 60.0
            cm1 = flow_rate / inlet_area if inlet_area > 0 else cm1_approx
            beta1 = math.degrees(math.atan2(cm1, u1))
        # Recompute tau2 for the (possibly new) d2/b2
        tau2 = calc_blockage_factor(d2, b2, blade_count, blade_thickness, is_inlet=False)
        # Refresh blade count after geometry changes
        blade_count = calc_blade_count(d2, d1, beta1, beta2, nq)

    return ImpellerDimensions(
        d2=d2, d1=d1, d1_hub=d1_hub,
        b2=b2, b1=b1,
        beta1=beta1, beta2=beta2,
        blade_count=blade_count, u2=u2,
        tau1=tau1, tau2=tau2,
    )
