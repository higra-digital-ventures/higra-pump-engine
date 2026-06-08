"""Radial inflow turbine (IFR) sizing — gas-path equivalent of Francis.

Computes preliminary rotor geometry and performance for radial inflow
turbines used in turbochargers, ORC expanders, and small gas turbines.
The method follows the Whitfield & Baines approach with Rodgers/Glassman
loss correlations.

Design philosophy:
    - Nozzle accelerates flow, rotor extracts work
    - U/C0 (velocity ratio) is the primary design parameter (~0.7 optimum)
    - Zero exit swirl is the default design target

References:
    - Whitfield, A. & Baines, N.C. (1990). Design of Radial Turbomachines.
    - Glassman, A.J. (1976). Turbine Design and Application (NASA SP-290).
    - Rodgers, C. (1987). Mainline performance prediction for radial
      inflow turbines (VKI Lecture Series).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field



@dataclass
class GasProps:
    """Ideal-gas working fluid properties."""

    name: str = "Air"
    gamma: float = 1.4
    R: float = 287.05  # J/(kg K)
    cp: float = 1004.5  # J/(kg K)

    @property
    def cv(self) -> float:
        """Specific heat at constant volume [J/(kg K)]."""
        return self.cp - self.R


@dataclass
class RadialTurbineResult:
    """Radial inflow turbine preliminary sizing result."""

    # Specific speed
    ns: float  # Specific speed (rad, W, m^3/s based)
    ns_dim: float  # Dimensional Ns (rpm, m^3/s, J/kg)

    # Rotor dimensions [m]
    d2: float  # Rotor inlet (tip) diameter
    d1: float  # Rotor exit (hub-to-shroud mean) diameter
    b2: float  # Rotor inlet blade width

    # Blade angles [deg]
    alpha2: float  # Nozzle exit / rotor inlet absolute flow angle
    beta2: float  # Rotor inlet relative blade angle
    beta1: float  # Rotor exit relative blade angle

    # Velocity ratio
    u_c0: float  # U2 / C0 (isentropic spouting velocity ratio)

    # Velocities [m/s]
    c0: float  # Isentropic spouting velocity
    u2: float  # Rotor tip speed
    cm2: float  # Meridional velocity at rotor inlet
    cm1: float  # Meridional velocity at rotor exit

    # Thermodynamic
    T0_in: float  # Total temperature at inlet [K]
    p0_in: float  # Total pressure at inlet [Pa]
    p_out: float  # Static pressure at outlet [Pa]
    pressure_ratio: float  # p0_in / p_out (total-to-static)

    # Loss breakdown (enthalpy loss coefficients, fraction of C0^2/2)
    loss_nozzle: float
    loss_rotor: float
    loss_tip_clearance: float
    loss_exit_ke: float
    loss_total: float

    # Efficiency
    eta_ts: float  # Total-to-static efficiency
    eta_tt: float  # Total-to-total efficiency

    # Power
    power: float  # [W]
    mass_flow: float  # [kg/s]
    rpm: float

    blade_count: int
    warnings: list[str] = field(default_factory=list)


def size_radial_turbine(
    P_total_in: float,
    T_total_in: float,
    p_out: float,
    mass_flow: float,
    rpm: float,
    gas_props: GasProps | None = None,
) -> RadialTurbineResult:
    """Preliminary sizing of a radial inflow turbine.

    Args:
        P_total_in: Inlet total pressure [Pa].
        T_total_in: Inlet total temperature [K].
        p_out: Outlet static pressure [Pa].
        mass_flow: Mass flow rate [kg/s].
        rpm: Rotational speed [rev/min].
        gas_props: Working fluid properties (defaults to air).

    Returns:
        RadialTurbineResult with geometry, losses, and efficiency.

    Raises:
        ValueError: If pressure ratio is less than 1 or inputs are non-physical.
    """
    gas = gas_props or GasProps()
    warnings: list[str] = []

    if P_total_in <= p_out:
        raise ValueError(
            f"Inlet total pressure ({P_total_in:.0f} Pa) must exceed "
            f"outlet static pressure ({p_out:.0f} Pa)."
        )
    if T_total_in <= 0 or mass_flow <= 0 or rpm <= 0:
        raise ValueError("T_total_in, mass_flow and rpm must be positive.")

    g = gas.gamma
    cp = gas.cp
    R = gas.R
    omega = 2.0 * math.pi * rpm / 60.0

    # --- Isentropic expansion ---
    pr = P_total_in / p_out  # total-to-static pressure ratio
    exp_ts = (g - 1.0) / g
    T_out_s = T_total_in * (1.0 / pr) ** exp_ts  # isentropic exit static T
    dh_s = cp * (T_total_in - T_out_s)  # isentropic enthalpy drop [J/kg]

    # Isentropic spouting velocity
    c0 = math.sqrt(2.0 * dh_s)

    # --- Optimal velocity ratio (Rodgers correlation) ---
    # U/C0 ~ 0.70 is optimum for radial inflow turbines
    u_c0_target = 0.70
    u2 = u_c0_target * c0

    # Rotor tip diameter
    d2 = 2.0 * u2 / omega

    # --- Nozzle exit / rotor inlet flow ---
    # Target alpha2 ~ 70-75 deg for good nozzle performance
    alpha2 = 72.0  # [deg] nozzle exit absolute angle
    alpha2_rad = math.radians(alpha2)

    # Absolute velocity at rotor inlet (from Euler and zero exit swirl)
    # With zero exit swirl: W_specific = u2 * cu2
    # cu2 = u2 (for radial blades at inlet, beta2 = 0 ideal)
    # Actual: cu2 from alpha2 and cm2
    # cm2/u2 ~ 0.2-0.35 (flow coefficient)
    phi2 = 0.28  # meridional flow coefficient cm2/u2
    cm2 = phi2 * u2
    cu2 = cm2 / math.tan(alpha2_rad) if alpha2_rad > 0.01 else u2
    c2 = math.sqrt(cm2**2 + cu2**2)

    # Relative velocity at rotor inlet
    wu2 = u2 - cu2
    w2 = math.sqrt(cm2**2 + wu2**2)
    beta2 = math.degrees(math.atan2(cm2, wu2))

    # Rotor inlet width
    rho2 = P_total_in / (R * T_total_in)  # approximate density at inlet
    # Correct for actual static conditions at station 2
    T2_s = T_total_in - c2**2 / (2.0 * cp)
    p2_s = P_total_in * (T2_s / T_total_in) ** (g / (g - 1.0))
    rho2 = p2_s / (R * T2_s) if T2_s > 0 else rho2
    b2 = mass_flow / (rho2 * math.pi * d2 * cm2) if (rho2 * d2 * cm2) > 0 else 0.01

    # --- Rotor exit ---
    # Exit diameter ratio D1/D2 ~ 0.55-0.70 (shroud diameter at exit)
    d1_d2 = 0.60
    d1 = d2 * d1_d2
    u1 = omega * d1 / 2.0

    # Zero exit swirl design: cu1 = 0
    # Euler work: w_specific = u2*cu2 - u1*cu1 = u2*cu2
    w_specific = u2 * cu2

    # Exit meridional velocity (continuity, approximate)
    # Annulus at exit: A1 = pi/4 * (d1_shroud^2 - d1_hub^2)
    hub_tip_exit = 0.30  # hub/shroud ratio at exit
    d1_hub = d1 * hub_tip_exit
    a1_exit = math.pi / 4.0 * (d1**2 * (1.0 - hub_tip_exit**2))
    # Exit density — use actual total temperature after work extraction
    # T0_exit = T0_in - w_specific / cp (energy balance)
    T0_exit = T_total_in - w_specific / cp
    # Exit static temperature (iterate: T1 = T0_exit - cm1^2/(2cp), start with T_out_s)
    T1_approx = max(T_out_s, T0_exit * 0.9)
    rho1 = p_out / (R * max(T1_approx, 100.0))
    cm1 = mass_flow / (rho1 * a1_exit) if (rho1 * a1_exit) > 0 else cm2 * 0.8
    # One Newton iteration for self-consistent T1
    T1_approx = T0_exit - cm1**2 / (2.0 * cp)
    T1_approx = max(T1_approx, T_out_s * 0.85)
    rho1 = p_out / (R * max(T1_approx, 100.0))
    cm1 = mass_flow / (rho1 * a1_exit) if (rho1 * a1_exit) > 0 else cm2 * 0.8

    # Exit blade angle
    w1 = math.sqrt(cm1**2 + u1**2)  # zero exit swirl: wu1 = u1
    beta1 = math.degrees(math.atan2(cm1, u1))

    # --- Loss model (all expressed as fraction of isentropic dh_s) ---
    # Nozzle loss: velocity coefficient phi_n ~ 0.97 => loss ~ 1 - phi_n^2
    phi_n = 0.97
    loss_nozzle = (1.0 - phi_n**2) * (c2**2 / 2.0) / dh_s if dh_s > 0 else 0.0

    # Rotor passage loss (Glassman/NASA SP-290)
    # K_p * (w1^2 + w2^2) / (2 * dh_s), K_p ~ 0.10-0.15
    k_rotor = 0.12
    loss_rotor = k_rotor * (w2**2 + w1**2) / (2.0 * dh_s) if dh_s > 0 else 0.0

    # Tip clearance loss (Rodgers: delta_eta ~ 0.93 * eps/b2)
    eps_b2 = 0.02  # clearance / blade width ratio
    loss_tip_clearance = 0.93 * eps_b2

    # Exit kinetic energy loss (total-to-static only)
    c1 = cm1  # zero swirl => c1 = cm1
    loss_exit_ke = c1**2 / (2.0 * dh_s) if dh_s > 0 else 0.0

    loss_total = loss_nozzle + loss_rotor + loss_tip_clearance + loss_exit_ke

    # --- Efficiency ---
    eta_ts = 1.0 - loss_total
    eta_ts = max(0.3, min(0.95, eta_ts))

    # Total-to-total: exclude exit KE loss
    eta_tt = eta_ts + loss_exit_ke
    eta_tt = max(eta_ts, min(0.96, eta_tt))

    # --- Power ---
    power = mass_flow * w_specific * eta_ts / max(u_c0_target**2 * c0**2 / (u2 * cu2), 1.0)
    # Simpler: P = m_dot * eta_ts * dh_s
    power = mass_flow * eta_ts * dh_s

    # --- Specific speed ---
    # Ns = omega * sqrt(Q_exit) / (dh_s)^0.75
    q_exit = mass_flow / rho1 if rho1 > 0 else mass_flow / 1.2
    ns = omega * math.sqrt(q_exit) / dh_s**0.75 if dh_s > 0 else 0.0

    # Dimensional Ns (rpm, m^3/s, J/kg based)
    ns_dim = rpm * math.sqrt(q_exit) / dh_s**0.75 if dh_s > 0 else 0.0

    # --- Blade count (Glassman) ---
    # Z = pi/30 * (110 - alpha2) * tan(alpha2)
    z_calc = math.pi / 30.0 * (110.0 - alpha2) * math.tan(alpha2_rad)
    blade_count = max(8, min(20, round(z_calc)))

    # --- Warnings ---
    if u_c0_target < 0.55 or u_c0_target > 0.85:
        warnings.append(f"U/C0={u_c0_target:.2f} outside optimal range [0.55, 0.85].")
    if beta2 < -20.0 or beta2 > 40.0:
        warnings.append(f"Rotor inlet relative angle beta2={beta2:.1f} deg unusual.")
    if d2 > 2.0:
        warnings.append(f"Large rotor diameter D2={d2*1000:.0f} mm. Check rpm.")
    if eta_ts < 0.70:
        warnings.append(f"Low total-to-static efficiency {eta_ts*100:.1f}%.")
    if pr > 6.0:
        warnings.append(f"High pressure ratio {pr:.1f}. Consider two-stage design.")

    return RadialTurbineResult(
        ns=ns,
        ns_dim=ns_dim,
        d2=d2,
        d1=d1,
        b2=b2,
        alpha2=alpha2,
        beta2=beta2,
        beta1=beta1,
        u_c0=u_c0_target,
        c0=c0,
        u2=u2,
        cm2=cm2,
        cm1=cm1,
        T0_in=T_total_in,
        p0_in=P_total_in,
        p_out=p_out,
        pressure_ratio=pr,
        loss_nozzle=loss_nozzle,
        loss_rotor=loss_rotor,
        loss_tip_clearance=loss_tip_clearance,
        loss_exit_ke=loss_exit_ke,
        loss_total=loss_total,
        eta_ts=eta_ts,
        eta_tt=eta_tt,
        power=power,
        mass_flow=mass_flow,
        rpm=rpm,
        blade_count=blade_count,
        warnings=warnings,
    )
