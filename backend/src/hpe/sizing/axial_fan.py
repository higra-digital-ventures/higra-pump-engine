"""Axial fan sizing module.

Preliminary sizing for axial-flow fans (ventilation, HVAC, industrial)
with fan-specific correlations including the Eck blade-number method,
NACA-based profile losses, and fan static efficiency.

Design approach:
    - Free-vortex design at mean radius
    - De Haller number check for diffusion limit
    - Eck correlation for blade count selection
    - Profile loss from NACA cascade data

References:
    - Eck, B. (2003). Fans: Design and Operation of Centrifugal,
      Axial-Flow and Cross-Flow Fans.
    - Wallis, R.A. (1983). Axial Flow Fans and Ducts.
    - Dixon & Hall (2014). Fluid Mech. & Thermo of Turbomachinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field



@dataclass
class AxialFanResult:
    """Axial fan preliminary sizing result."""

    # Dimensions [m]
    d_tip: float
    d_hub: float
    hub_tip_ratio: float
    blade_height: float
    d_mean: float

    # Blade geometry
    blade_count: int
    chord: float  # [m]
    solidity: float  # chord / pitch
    stagger_angle: float  # [deg]

    # Angles at mean radius [deg]
    beta1_mean: float
    beta2_mean: float
    alpha1_mean: float
    alpha2_mean: float

    # Performance
    de_haller: float
    diffusion_factor: float  # Lieblein D-factor
    fan_static_efficiency: float
    fan_total_efficiency: float
    power: float  # [W]

    # Flow
    flow_rate: float  # [m^3/s]
    total_pressure_rise: float  # [Pa]
    static_pressure_rise: float  # [Pa]
    axial_velocity: float  # [m/s]
    tip_speed: float  # [m/s]
    specific_speed: float  # dimensionless Ns
    flow_coefficient: float  # phi = cm / u_tip
    pressure_coefficient: float  # psi = dp / (0.5 * rho * u_tip^2)

    # Loss breakdown
    loss_profile: float  # NACA profile loss coefficient
    loss_secondary: float  # secondary flow loss
    loss_tip_clearance: float  # tip clearance loss
    loss_annulus: float  # annulus wall loss

    warnings: list[str] = field(default_factory=list)


def size_axial_fan(
    flow_rate: float,
    total_pressure_rise: float,
    rpm: float,
    hub_tip_ratio: float = 0.5,
    rho: float = 1.2,
) -> AxialFanResult:
    """Preliminary sizing of an axial fan.

    Args:
        flow_rate: Volume flow rate Q [m^3/s].
        total_pressure_rise: Total pressure rise delta_p0 [Pa].
        rpm: Rotational speed [rev/min].
        hub_tip_ratio: Hub-to-tip diameter ratio (0.3-0.8).
        rho: Air density [kg/m^3].

    Returns:
        AxialFanResult with geometry, losses, and efficiency.

    Raises:
        ValueError: If inputs are non-physical.
    """
    if flow_rate <= 0 or total_pressure_rise <= 0 or rpm <= 0:
        raise ValueError("flow_rate, total_pressure_rise, and rpm must be positive.")
    if not 0.2 <= hub_tip_ratio <= 0.9:
        raise ValueError(f"hub_tip_ratio={hub_tip_ratio} outside valid range [0.2, 0.9].")

    warnings: list[str] = []
    omega = 2.0 * math.pi * rpm / 60.0

    # --- Fan specific speed (dimensionless) ---
    # Ns = omega * sqrt(Q) / (dp/rho)^0.75
    dp_rho = total_pressure_rise / rho
    ns = omega * math.sqrt(flow_rate) / dp_rho**0.75 if dp_rho > 0 else 0.0

    # --- Tip diameter from flow and pressure ---
    # Target flow coefficient phi = cm/u_tip ~ 0.2-0.5
    # Target pressure coefficient psi = dp/(0.5*rho*u_tip^2) ~ 0.1-0.6
    psi_target = 0.35  # moderate loading
    u_tip = math.sqrt(total_pressure_rise / (0.5 * rho * psi_target))
    d_tip = 2.0 * u_tip / omega

    d_hub = hub_tip_ratio * d_tip
    d_mean = (d_tip + d_hub) / 2.0
    blade_height = (d_tip - d_hub) / 2.0

    # Annulus area and axial velocity
    a_annulus = math.pi / 4.0 * (d_tip**2 - d_hub**2)
    cm = flow_rate / a_annulus if a_annulus > 0 else 1.0

    phi = cm / u_tip if u_tip > 0 else 0.0
    psi_actual = total_pressure_rise / (0.5 * rho * u_tip**2) if u_tip > 0 else 0.0

    # --- Velocity triangles at mean radius ---
    u_mean = omega * d_mean / 2.0
    # Euler: dp_total = rho * u_mean * delta_cu * eta_h
    eta_h_est = 0.85  # initial estimate
    delta_cu = total_pressure_rise / (rho * u_mean * eta_h_est) if (rho * u_mean) > 0 else 0.0

    # Assume axial inlet (alpha1 = 90, cu1 = 0)
    cu1 = 0.0
    cu2 = delta_cu

    # Absolute velocities
    c1 = math.sqrt(cm**2 + cu1**2)
    c2 = math.sqrt(cm**2 + cu2**2)

    # Relative velocities
    wu1 = u_mean - cu1
    wu2 = u_mean - cu2
    w1 = math.sqrt(cm**2 + wu1**2)
    w2 = math.sqrt(cm**2 + wu2**2)

    # Angles
    beta1 = math.degrees(math.atan2(cm, wu1))
    beta2 = math.degrees(math.atan2(cm, wu2))
    alpha1 = 90.0  # axial inlet
    alpha2 = math.degrees(math.atan2(cm, cu2)) if cu2 > 1e-6 else 90.0

    # --- De Haller number ---
    de_haller = w2 / w1 if w1 > 0 else 1.0
    if de_haller < 0.72:
        warnings.append(f"De Haller number {de_haller:.3f} < 0.72: risk of blade stall.")

    # --- Blade count (Eck correlation) ---
    # Z_eck = 2*pi * cos(beta_mean) / ((1 - hub_tip_ratio) * mean_turning)
    beta_mean_rad = math.radians((beta1 + beta2) / 2.0)
    turning = abs(beta1 - beta2)
    turning_rad = math.radians(turning) if turning > 0 else 0.1
    z_eck = (2.0 * math.pi * math.cos(beta_mean_rad)
             / ((1.0 - hub_tip_ratio) * turning_rad))
    blade_count = max(4, min(24, round(z_eck)))

    # --- Solidity and chord ---
    pitch_mean = math.pi * d_mean / blade_count
    # Lieblein D-factor target ~ 0.45
    d_target = 0.45
    if w1 > 0:
        sigma_req = abs(delta_cu) / (2.0 * w1 * max(d_target - 1.0 + w2 / w1, 0.01))
        sigma_req = max(0.4, min(2.0, sigma_req))
    else:
        sigma_req = 1.0

    chord = sigma_req * pitch_mean
    solidity = sigma_req

    # Stagger angle
    stagger = (beta1 + beta2) / 2.0

    # Diffusion factor (Lieblein)
    diff_factor = (1.0 - w2 / w1 + abs(delta_cu) / (2.0 * solidity * w1)
                   if w1 > 0 else 0.0)
    if diff_factor > 0.6:
        warnings.append(f"Lieblein D-factor {diff_factor:.3f} > 0.6: high loading.")

    # --- Loss model ---
    # Profile loss (NACA cascade-based)
    # omega_p = 2*sigma * (0.006 + 0.2*(Dsurf)^2) -- simplified NACA
    d_surf = diff_factor  # surface diffusion ~ Lieblein factor
    omega_profile = 2.0 * solidity * (0.006 + 0.2 * d_surf**2)
    loss_profile = omega_profile * w1**2 / (2.0 * total_pressure_rise / rho) if total_pressure_rise > 0 else 0.0

    # Secondary flow loss
    # omega_s = 0.018 * Cl^2 / (solidity * aspect_ratio)
    aspect_ratio = blade_height / chord if chord > 0 else 3.0
    cl = 2.0 * abs(delta_cu) * pitch_mean / (w1 * chord) if (w1 * chord) > 0 else 0.5
    omega_secondary = 0.018 * cl**2 / (solidity * max(aspect_ratio, 0.5))
    loss_secondary = omega_secondary * w1**2 / (2.0 * dp_rho) if dp_rho > 0 else 0.0

    # Tip clearance loss
    eps_h = 0.01  # clearance / blade height ratio
    loss_tip_clearance = 0.3 * eps_h * cl / solidity

    # Annulus (endwall) loss
    loss_annulus = 0.02 * (1.0 + d_hub / d_tip) / aspect_ratio if aspect_ratio > 0 else 0.02

    loss_total = loss_profile + loss_secondary + loss_tip_clearance + loss_annulus

    # --- Efficiency ---
    fan_total_efficiency = max(0.4, min(0.92, 1.0 - loss_total))

    # Static pressure rise
    # dp_static = dp_total - 0.5*rho*(c2^2 - c1^2)
    dp_dynamic = 0.5 * rho * (c2**2 - c1**2)
    dp_static = total_pressure_rise - dp_dynamic

    fan_static_efficiency = (fan_total_efficiency * dp_static / total_pressure_rise
                             if total_pressure_rise > 0 else fan_total_efficiency * 0.8)
    fan_static_efficiency = max(0.3, min(0.90, fan_static_efficiency))

    # Power
    power = flow_rate * total_pressure_rise / fan_total_efficiency if fan_total_efficiency > 0 else 0.0

    # --- Warnings ---
    if ns < 1.0:
        warnings.append(f"Low specific speed Ns={ns:.2f}. Consider centrifugal fan.")
    if ns > 6.0:
        warnings.append(f"High specific speed Ns={ns:.2f}. Check design point.")
    if phi < 0.1:
        warnings.append(f"Low flow coefficient phi={phi:.3f}.")
    if u_tip > 120.0:
        warnings.append(f"High tip speed {u_tip:.0f} m/s. Check noise constraints.")

    return AxialFanResult(
        d_tip=d_tip,
        d_hub=d_hub,
        hub_tip_ratio=hub_tip_ratio,
        blade_height=blade_height,
        d_mean=d_mean,
        blade_count=blade_count,
        chord=chord,
        solidity=solidity,
        stagger_angle=stagger,
        beta1_mean=beta1,
        beta2_mean=beta2,
        alpha1_mean=alpha1,
        alpha2_mean=alpha2,
        de_haller=de_haller,
        diffusion_factor=diff_factor,
        fan_static_efficiency=fan_static_efficiency,
        fan_total_efficiency=fan_total_efficiency,
        power=power,
        flow_rate=flow_rate,
        total_pressure_rise=total_pressure_rise,
        static_pressure_rise=dp_static,
        axial_velocity=cm,
        tip_speed=u_tip,
        specific_speed=ns,
        flow_coefficient=phi,
        pressure_coefficient=psi_actual,
        loss_profile=loss_profile,
        loss_secondary=loss_secondary,
        loss_tip_clearance=loss_tip_clearance,
        loss_annulus=loss_annulus,
        warnings=warnings,
    )
