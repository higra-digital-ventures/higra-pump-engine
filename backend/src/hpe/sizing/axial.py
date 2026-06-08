"""Axial and mixed-flow pump/fan/compressor sizing.

Extends HPE beyond centrifugal machines to axial and mixed-flow
configurations. Uses the same OperatingPoint → SizingResult pipeline.

Design method:
    - Free-vortex design for axial machines
    - Hub-to-tip ratio from specific speed correlation
    - Blade number from Zweifel criterion
    - De Haller number check (w2/w1 > 0.72)

References:
    - Dixon & Hall (2014). Fluid Mechanics & Thermo of Turbomachinery.
    - Cumpsty (2004). Compressor Aerodynamics.
    - Pfleiderer & Petermann (2005). Strömungsmaschinen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpe.core.models import G, OperatingPoint
from hpe.sizing.efficiency import estimate_all_efficiencies
from hpe.sizing.specific_speed import calc_specific_speed


@dataclass
class AxialSizingResult:
    """Axial/mixed-flow sizing output."""

    # Specific speed
    nq: float
    ns: float

    # Dimensions
    d_tip: float  # Tip diameter [m]
    d_hub: float  # Hub diameter [m]
    hub_tip_ratio: float  # d_hub / d_tip
    blade_height: float  # (d_tip - d_hub) / 2 [m]
    d_mean: float  # Mean diameter [m]

    # Blade
    blade_count: int
    chord: float  # Blade chord [m]
    solidity: float  # chord / pitch
    stagger_angle: float  # [deg]

    # Angles at mean radius [deg]
    beta1_mean: float  # Inlet relative angle
    beta2_mean: float  # Outlet relative angle
    alpha1_mean: float  # Inlet absolute angle
    alpha2_mean: float  # Outlet absolute angle

    # Performance
    de_haller: float  # w2/w1 (should be > 0.72)
    diffusion_factor: float  # Lieblein D-factor
    estimated_efficiency: float
    estimated_power: float  # [W]

    # Flow
    axial_velocity: float  # cm [m/s]
    reaction_degree: float  # R (0.5 = 50% reaction)

    warnings: list[str]


def size_axial(
    op: OperatingPoint,
    reaction: float = 0.5,
    hub_tip_ratio: float | None = None,
) -> AxialSizingResult:
    """Size an axial pump, fan, or compressor stage.

    Args:
        op: Operating point (Q, H, rpm).
        reaction: Degree of reaction (0.5 = symmetric, typical for pumps).
        hub_tip_ratio: Hub-to-tip ratio. If None, estimated from Nq.

    Returns:
        AxialSizingResult with all dimensions and performance.
    """
    warnings: list[str] = []

    ns, nq = calc_specific_speed(op.flow_rate, op.head, op.rpm)
    omega = 2.0 * math.pi * op.rpm / 60.0

    # Efficiency
    eta_h, eta_v, eta_m, eta_total = estimate_all_efficiencies(op.flow_rate, nq)

    # Hub-to-tip ratio from Nq (Pfleiderer correlation)
    if hub_tip_ratio is None:
        if nq < 100:
            hub_tip_ratio = 0.3 + 0.1 * (nq / 100)
        elif nq < 200:
            hub_tip_ratio = 0.4 + 0.15 * ((nq - 100) / 100)
        else:
            hub_tip_ratio = 0.55 + 0.1 * min((nq - 200) / 100, 1.0)
        hub_tip_ratio = max(0.3, min(0.8, hub_tip_ratio))

    # Euler head required
    h_euler = op.head / eta_h

    # Tip diameter from continuity: Q = cm * pi/4 * (Dt^2 - Dh^2)
    # Assume cm/u_tip ratio ~ 0.3-0.5 (flow coefficient)
    phi = 0.35  # Flow coefficient cm/u_tip (typical)

    # u_tip from head: H_euler = psi * u_tip^2 / g
    # psi (head coefficient) for axial: 0.3-0.5
    psi = 0.35  # Head coefficient

    u_tip = math.sqrt(h_euler * G / psi)
    d_tip = 2.0 * u_tip / omega

    d_hub = hub_tip_ratio * d_tip
    d_mean = (d_tip + d_hub) / 2.0

    # Axial velocity
    annulus_area = math.pi / 4.0 * (d_tip**2 - d_hub**2)
    cm = op.flow_rate / annulus_area if annulus_area > 0 else 0

    # Velocities at mean radius
    u_mean = omega * d_mean / 2.0

    # Euler: Delta_cu = g * H_euler / u_mean
    delta_cu = G * h_euler / u_mean if u_mean > 0 else 0

    # With reaction R: cu1 = u_mean * (1-R) - delta_cu/2, cu2 = cu1 + delta_cu
    # For R=0.5: cu1 = -delta_cu/2, cu2 = +delta_cu/2
    cu1 = u_mean * (1 - reaction) - delta_cu / 2.0
    cu2 = cu1 + delta_cu

    # Velocity triangles at mean
    c1 = math.sqrt(cm**2 + cu1**2)
    c2 = math.sqrt(cm**2 + cu2**2)

    wu1 = u_mean - cu1
    wu2 = u_mean - cu2
    w1 = math.sqrt(cm**2 + wu1**2)
    w2 = math.sqrt(cm**2 + wu2**2)

    beta1 = math.degrees(math.atan2(cm, wu1))
    beta2 = math.degrees(math.atan2(cm, wu2))
    alpha1 = math.degrees(math.atan2(cm, cu1)) if abs(cu1) > 1e-6 else 90.0
    alpha2 = math.degrees(math.atan2(cm, cu2)) if abs(cu2) > 1e-6 else 90.0

    # De Haller number
    de_haller = w2 / w1 if w1 > 0 else 1.0
    if de_haller < 0.72:
        warnings.append(f"De Haller number {de_haller:.2f} < 0.72: risk of separation")

    # Blade count (Zweifel criterion based)
    pitch_mean = math.pi * d_mean / 14  # Start with ~14 blades
    blade_height = (d_tip - d_hub) / 2.0

    # Solidity from diffusion factor target ~0.45
    # D = 1 - w2/w1 + delta_cu / (2*sigma*w1)
    d_target = 0.45
    if w1 > 0:
        sigma_target = delta_cu / (2.0 * w1 * (d_target - 1.0 + w2 / w1))
        sigma_target = max(0.5, min(2.0, abs(sigma_target)))
    else:
        sigma_target = 1.0

    # Blade count from solidity
    chord_est = sigma_target * math.pi * d_mean / 14
    blade_count = max(6, min(30, round(math.pi * d_mean * sigma_target / chord_est)))

    pitch = math.pi * d_mean / blade_count
    chord = sigma_target * pitch
    solidity = chord / pitch

    # Stagger angle
    stagger = math.degrees(math.atan2(cm, (wu1 + wu2) / 2.0))

    # Diffusion factor (Lieblein)
    diff_factor = 1.0 - w2 / w1 + abs(delta_cu) / (2.0 * solidity * w1) if w1 > 0 else 0
    if diff_factor > 0.6:
        warnings.append(f"Diffusion factor {diff_factor:.2f} > 0.6: risk of stall")

    # Power
    power = op.fluid_density * G * op.flow_rate * op.head / eta_total

    if nq < 70:
        warnings.append(f"Nq={nq:.0f} is low for axial design. Consider mixed-flow or centrifugal.")

    return AxialSizingResult(
        nq=nq, ns=ns,
        d_tip=d_tip, d_hub=d_hub, hub_tip_ratio=hub_tip_ratio,
        blade_height=blade_height, d_mean=d_mean,
        blade_count=blade_count, chord=chord,
        solidity=solidity, stagger_angle=stagger,
        beta1_mean=beta1, beta2_mean=beta2,
        alpha1_mean=alpha1, alpha2_mean=alpha2,
        de_haller=de_haller, diffusion_factor=diff_factor,
        estimated_efficiency=eta_total, estimated_power=power,
        axial_velocity=cm, reaction_degree=reaction,
        warnings=warnings,
    )
