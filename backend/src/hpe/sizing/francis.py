"""Francis turbine sizing correlations.

Extends the meanline sizing for Francis turbines and pump-turbines.
Key differences from pumps:
- Energy is extracted (turbine mode): H_euler is negative from flow perspective
- Specific speed uses power-based definition for turbines
- Guide vanes control flow, not impeller inlet angle
- Draft tube recovers kinetic energy at runner outlet

References:
    - Brekke, H. (2001). Hydraulic Turbines — Design, Erection and Operation.
    - IEC 60193: Hydraulic turbines, storage pumps and pump-turbines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpe.core.models import G


@dataclass
class FrancisSizing:
    """Francis turbine preliminary sizing result."""

    # Specific speed
    ns_power: float  # Ns (power-based) = n * sqrt(P) / H^(5/4)
    nq: float  # Nq (flow-based, same as pumps)

    # Runner dimensions
    d1: float  # Runner inlet diameter [m]
    d2: float  # Runner outlet diameter [m]
    b0: float  # Guide vane height (distributor width) [m]
    d_draft: float  # Draft tube inlet diameter [m]

    # Angles
    beta1: float  # Runner inlet blade angle [deg]
    beta2: float  # Runner outlet blade angle [deg]
    alpha1: float  # Guide vane outlet angle (absolute flow angle) [deg]

    # Performance
    estimated_efficiency: float
    estimated_power: float  # [W]
    runner_speed: float  # u1 peripheral velocity [m/s]

    blade_count: int


def size_francis(
    flow_rate: float,
    head: float,
    rpm: float,
    rho: float = 998.2,
) -> FrancisSizing:
    """Preliminary sizing of a Francis turbine runner.

    Args:
        flow_rate: Q [m^3/s].
        head: Net head H [m].
        rpm: Rotational speed [rev/min].
        rho: Fluid density [kg/m^3].

    Returns:
        FrancisSizing with all preliminary dimensions.
    """
    # Specific speeds
    nq = rpm * math.sqrt(flow_rate) / head**0.75

    # Efficiency estimation (Francis turbines: 90-95% at BEP)
    eta = _francis_efficiency(nq, flow_rate)

    # Power
    power = eta * rho * G * flow_rate * head

    # Power-based specific speed
    ns_power = rpm * math.sqrt(power) / (rho * G * head) ** (5.0 / 4.0)

    # Runner inlet diameter D1
    # ku = u1 / sqrt(2gH), typical 0.62-0.78 for Francis
    ku = _francis_ku(nq)
    u1 = ku * math.sqrt(2.0 * G * head)
    d1 = 60.0 * u1 / (math.pi * rpm)

    # Runner outlet diameter D2
    # D2/D1 ratio depends on Nq (Brekke)
    d2_d1 = _francis_d2_d1_ratio(nq)
    d2 = d1 * d2_d1

    # Guide vane height b0
    # b0/D1 ratio from specific speed
    b0_d1 = _francis_b0_d1_ratio(nq)
    b0 = d1 * b0_d1

    # Draft tube diameter
    d_draft = d2 * 1.05

    # Blade angles
    # Alpha1 (guide vane exit / runner inlet flow angle)
    cm1 = flow_rate / (math.pi * d1 * b0)  # Meridional velocity at inlet
    cu1 = G * head * eta / u1  # From Euler: P = rho*Q*(u1*cu1)
    alpha1 = math.degrees(math.atan2(cm1, cu1))

    beta1 = math.degrees(math.atan2(cm1, u1 - cu1))

    # Beta2 (runner outlet)
    u2 = math.pi * d2 * rpm / 60.0
    a_out = math.pi / 4.0 * d2**2  # Approximate outlet area
    cm2 = flow_rate / a_out
    beta2 = math.degrees(math.atan2(cm2, u2))

    # Blade count (Francis: 13-17 typical)
    blade_count = _francis_blade_count(nq)

    return FrancisSizing(
        ns_power=ns_power,
        nq=nq,
        d1=d1,
        d2=d2,
        b0=b0,
        d_draft=d_draft,
        beta1=beta1,
        beta2=beta2,
        alpha1=alpha1,
        estimated_efficiency=eta,
        estimated_power=power,
        runner_speed=u1,
        blade_count=blade_count,
    )


def _francis_efficiency(nq: float, flow_rate: float) -> float:
    """Estimate Francis turbine efficiency."""
    # Peak efficiency at Nq ~ 60-80
    eta_base = 0.93
    nq_opt = 70.0
    nq_penalty = 0.001 * ((nq - nq_opt) / nq_opt) ** 2

    # Size correction (larger = more efficient)
    size_bonus = 0.01 * min(1.0, flow_rate / 1.0)

    return max(0.80, min(0.96, eta_base - nq_penalty + size_bonus))


def _francis_ku(nq: float) -> float:
    """Peripheral velocity coefficient ku = u1/sqrt(2gH)."""
    if nq < 40:
        return 0.62
    elif nq < 80:
        return 0.62 + 0.004 * (nq - 40)
    else:
        return 0.78 + 0.001 * (nq - 80)


def _francis_d2_d1_ratio(nq: float) -> float:
    """Runner outlet/inlet diameter ratio."""
    if nq < 40:
        return 0.65
    elif nq < 100:
        return 0.65 + 0.003 * (nq - 40)
    else:
        return 0.83


def _francis_b0_d1_ratio(nq: float) -> float:
    """Guide vane height to runner diameter ratio."""
    if nq < 30:
        return 0.08
    elif nq < 80:
        return 0.08 + 0.004 * (nq - 30)
    else:
        return 0.28 + 0.002 * (nq - 80)


def _francis_blade_count(nq: float) -> int:
    """Number of runner blades for Francis turbine."""
    if nq < 40:
        return 17
    elif nq < 70:
        return 15
    elif nq < 120:
        return 13
    else:
        return 11
