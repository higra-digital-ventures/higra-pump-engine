"""Sirocco (forward-curved centrifugal) fan sizing module.

Forward-curved fans have beta2 > 90 deg, generating high flow rates
at low rotational speeds. Typical applications: HVAC air handlers,
automotive blowers, small ventilation units.

Characteristics:
    - Forward-curved blades: beta2 = 120-160 deg
    - Low specific speed (Ns ~ 0.3-1.2)
    - Efficiency typically 50-65% (lower than backward-curved)
    - Wide, shallow impeller with many short blades (24-64)
    - Scroll (volute) housing required for pressure recovery

References:
    - Eck, B. (2003). Fans: Design and Operation.
    - Bleier, F.P. (1998). Fan Handbook.
    - ASHRAE Handbook — HVAC Systems and Equipment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field



@dataclass
class SiroccoFanResult:
    """Forward-curved (sirocco) fan preliminary sizing result."""

    # Impeller dimensions [m]
    d2: float  # Outer (tip) diameter
    d1: float  # Inner (inlet eye) diameter
    b2: float  # Impeller width at outlet
    d1_d2: float  # Diameter ratio

    # Blade geometry
    blade_count: int
    beta2: float  # Outlet blade angle [deg] (> 90)
    beta1: float  # Inlet blade angle [deg]
    blade_chord: float  # [m]

    # Scroll housing
    scroll_width: float  # [m]
    scroll_d_outer: float  # Scroll outer diameter [m]
    cutoff_clearance: float  # Gap between cutoff and impeller [m]

    # Performance
    flow_rate: float  # [m^3/s]
    static_pressure: float  # [Pa]
    total_pressure: float  # [Pa]
    power: float  # [W]
    fan_static_efficiency: float
    fan_total_efficiency: float
    tip_speed: float  # [m/s]
    rpm: float

    # Flow details
    specific_speed: float  # dimensionless
    flow_coefficient: float  # phi = Q / (u2 * D2 * b2)
    pressure_coefficient: float  # psi = dp / (0.5 * rho * u2^2)

    # Velocities [m/s]
    cm2: float  # Meridional velocity at outlet
    cu2: float  # Tangential velocity at outlet
    c2: float  # Absolute velocity at outlet
    w2: float  # Relative velocity at outlet

    warnings: list[str] = field(default_factory=list)


def size_sirocco_fan(
    flow_rate: float,
    static_pressure: float,
    rpm: float,
    rho: float = 1.2,
) -> SiroccoFanResult:
    """Preliminary sizing of a forward-curved (sirocco) fan.

    The fan static pressure includes the scroll (volute) pressure recovery.
    Forward-curved fans rely heavily on the scroll to convert the high
    impeller-exit dynamic pressure into static pressure.

    Args:
        flow_rate: Volume flow rate Q [m^3/s].
        static_pressure: Required fan static pressure rise [Pa].
        rpm: Rotational speed [rev/min].
        rho: Air density [kg/m^3].

    Returns:
        SiroccoFanResult with impeller, scroll, and performance data.

    Raises:
        ValueError: If inputs are non-physical.
    """
    if flow_rate <= 0 or static_pressure <= 0 or rpm <= 0:
        raise ValueError("flow_rate, static_pressure, and rpm must be positive.")

    warnings: list[str] = []
    omega = 2.0 * math.pi * rpm / 60.0

    # --- Specific speed ---
    dp_rho = static_pressure / rho
    ns = omega * math.sqrt(flow_rate) / dp_rho**0.75 if dp_rho > 0 else 0.0

    if ns > 2.0:
        warnings.append(
            f"Specific speed Ns={ns:.2f} is high for sirocco fan. "
            "Consider backward-curved or axial."
        )

    # --- Forward-curved blade angle ---
    beta2 = 140.0  # typical forward-curved angle [deg]
    beta2_rad = math.radians(beta2)

    # --- Sizing approach ---
    # For forward-curved fans, the Euler head coefficient
    # psi_euler = 2*cu2/u2 is high (3-5) because cu2 > u2.
    # The fan total efficiency is ~55-65%.
    # Fan total pressure = eta_t * dp_euler = eta_t * rho * u2 * cu2.
    # Fan static pressure = total_pressure - 0.5*rho*c_outlet^2
    # where c_outlet is the *duct* exit velocity (much less than c2
    # because the scroll decelerates the flow).
    #
    # Empirical: for a well-designed sirocco fan,
    #   fan_static_pressure ~ 0.55 * rho * u2^2   (Bleier)
    # This gives us u2 directly from the required static pressure.

    k_ps = 0.55  # empirical static-pressure coefficient (Bleier)
    u2 = math.sqrt(static_pressure / (rho * k_ps))
    d2 = 2.0 * u2 / omega

    # --- Diameter ratio ---
    d1_d2 = 0.80
    d1 = d2 * d1_d2

    # --- Impeller width ---
    # Flow coefficient phi = cm2/u2 ~ 0.2-0.35 for sirocco
    phi_target = 0.30
    cm2 = phi_target * u2
    b2 = flow_rate / (math.pi * d2 * cm2) if (d2 * cm2) > 0 else 0.05

    # --- Velocity triangle at outlet ---
    # Forward curved: cu2 = u2 - cm2/tan(beta2), tan(beta2) < 0 => cu2 > u2
    cu2 = u2 - cm2 / math.tan(beta2_rad)
    c2 = math.sqrt(cm2**2 + cu2**2)
    w2 = math.sqrt(cm2**2 + (u2 - cu2)**2)

    # --- Inlet blade angle ---
    u1 = omega * d1 / 2.0
    cm1 = flow_rate / (math.pi * d1 * b2) if (d1 * b2) > 0 else cm2
    beta1 = math.degrees(math.atan2(cm1, u1))

    # --- Blade count ---
    # Sirocco fans: 24-64 short, closely spaced blades
    # Simple rule: Z ~ pi*D2 / (2 * blade_chord), with chord ~ 0.8*(D2-D1)/2
    chord_approx = 0.8 * (d2 - d1) / 2.0
    if chord_approx > 0:
        z_calc = math.pi * d2 / (2.0 * chord_approx)
    else:
        z_calc = 36
    blade_count = max(24, min(64, round(z_calc / 2.0) * 2))

    blade_chord = math.pi * (d2 + d1) / (2.0 * blade_count)

    # --- Actual pressures (Euler-based) ---
    dp_euler = rho * u2 * cu2

    # Efficiency and total pressure
    eta_total = 0.60  # typical forward-curved fan total efficiency
    total_pressure = dp_euler * eta_total

    # Fan static pressure = total - exit_dynamic_at_duct
    # The scroll decelerates c2 to a duct velocity c_duct
    # Scroll pressure recovery coefficient ~ 0.4-0.6
    c_duct = flow_rate / (b2 * d2 * 0.5) if (b2 * d2) > 0 else cm2
    dp_duct_dynamic = 0.5 * rho * c_duct**2
    dp_static = total_pressure - dp_duct_dynamic
    dp_static = max(dp_static, total_pressure * 0.5)  # physical floor

    # --- Scroll housing ---
    scroll_width = b2 * 1.1
    scroll_d_outer = d2 * 1.6
    cutoff_clearance = d2 * 0.05

    # --- Efficiency ---
    fan_total_efficiency = max(0.45, min(0.65, eta_total))
    fan_static_efficiency = (fan_total_efficiency * dp_static / total_pressure
                             if total_pressure > 0 else 0.40)
    fan_static_efficiency = max(0.30, min(0.60, fan_static_efficiency))

    # --- Power ---
    power = (flow_rate * total_pressure / fan_total_efficiency
             if fan_total_efficiency > 0 else 0.0)

    # --- Flow coefficients ---
    flow_coeff = flow_rate / (u2 * d2 * b2) if (u2 * d2 * b2) > 0 else 0.0
    psi_actual = total_pressure / (0.5 * rho * u2**2) if u2 > 0 else 0.0

    # --- Warnings ---
    if fan_total_efficiency < 0.50:
        warnings.append(
            f"Total efficiency {fan_total_efficiency*100:.0f}% is low even for sirocco fan."
        )
    if u2 > 40.0:
        warnings.append(f"Tip speed {u2:.0f} m/s is high for sirocco fan. Check noise.")
    if b2 / d2 > 1.5:
        warnings.append(f"Very wide impeller b2/D2={b2/d2:.2f}. Consider double-width.")
    if cu2 / u2 > 2.5:
        warnings.append(f"Very high cu2/u2={cu2/u2:.2f}. Scroll losses will be large.")

    return SiroccoFanResult(
        d2=d2,
        d1=d1,
        b2=b2,
        d1_d2=d1_d2,
        blade_count=blade_count,
        beta2=beta2,
        beta1=beta1,
        blade_chord=blade_chord,
        scroll_width=scroll_width,
        scroll_d_outer=scroll_d_outer,
        cutoff_clearance=cutoff_clearance,
        flow_rate=flow_rate,
        static_pressure=dp_static,
        total_pressure=total_pressure,
        power=power,
        fan_static_efficiency=fan_static_efficiency,
        fan_total_efficiency=fan_total_efficiency,
        tip_speed=u2,
        rpm=rpm,
        specific_speed=ns,
        flow_coefficient=flow_coeff,
        pressure_coefficient=psi_actual,
        cm2=cm2,
        cu2=cu2,
        c2=c2,
        w2=w2,
        warnings=warnings,
    )
