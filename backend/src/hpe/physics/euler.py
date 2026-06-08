"""Off-design Euler head calculation.

Reuses velocity triangle functions from the sizing module but with
variable flow rate Q to predict theoretical head at any operating point.

At off-design conditions:
- u (peripheral velocity) stays constant (same geometry and RPM)
- cm (meridional velocity) changes proportionally with Q
- cu2 changes because cm2 changes (blade angle beta2 is fixed)
- The Euler head H = (u2*cu2 - u1*cu1) / g changes accordingly
"""

from __future__ import annotations

from hpe.core.models import SizingResult, VelocityTriangle
from hpe.sizing.velocity_triangles import (
    calc_euler_head,
    calc_inlet_triangle,
    calc_outlet_triangle,
)


def calc_off_design_triangles(
    sizing: SizingResult,
    q_actual: float,
) -> tuple[VelocityTriangle, VelocityTriangle]:
    """Calculate velocity triangles at an off-design flow rate.

    The impeller geometry (D1, D2, b1, b2, beta2, blade_count) is fixed
    from the sizing result. Only the flow rate changes.

    Args:
        sizing: SizingResult from the design-point sizing.
        q_actual: Actual volumetric flow rate [m^3/s].

    Returns:
        Tuple of (inlet_triangle, outlet_triangle).
    """
    mp = sizing.meridional_profile
    d1 = sizing.impeller_d1
    b1 = mp.get("b1", sizing.impeller_b2 * 1.2)

    # Extract RPM from velocity triangles
    # u2 = pi * D2 * rpm / 60 => rpm = 60 * u2 / (pi * D2)
    u2 = sizing.velocity_triangles["outlet"]["u"]
    import math
    rpm = 60.0 * u2 / (math.pi * sizing.impeller_d2)

    tri_in = calc_inlet_triangle(
        d1=d1,
        b1=b1,
        flow_rate=q_actual,
        rpm=rpm,
    )

    tri_out = calc_outlet_triangle(
        d2=sizing.impeller_d2,
        b2=sizing.impeller_b2,
        flow_rate=q_actual,
        rpm=rpm,
        beta2=sizing.beta2,
        blade_count=sizing.blade_count,
    )

    return tri_in, tri_out


def calc_off_design_euler_head(
    sizing: SizingResult,
    q_actual: float,
) -> float:
    """Calculate theoretical Euler head at an off-design flow rate.

    H_euler = (u2 * cu2 - u1 * cu1) / g

    At off-design:
    - cm changes with Q (more/less flow through same area)
    - cu2 changes because cu2 = u2 - cm2/tan(beta2) and cm2 changed
    - Result: Euler head decreases with increasing Q (for backward-curved blades)

    Args:
        sizing: SizingResult from design-point sizing.
        q_actual: Actual flow rate [m^3/s].

    Returns:
        Euler head [m] at the given flow rate.
    """
    tri_in, tri_out = calc_off_design_triangles(sizing, q_actual)
    return calc_euler_head(tri_in, tri_out)


def get_design_flow_rate(sizing: SizingResult) -> float:
    """Extract the design-point flow rate from a SizingResult.

    Uses the relationship: Q = cm2 * pi * D2 * b2 * blockage

    Args:
        sizing: SizingResult.

    Returns:
        Design flow rate Q [m^3/s].
    """
    import math
    cm2 = sizing.velocity_triangles["outlet"]["cm"]
    blockage = 0.88  # Same default as velocity_triangles module
    return cm2 * math.pi * sizing.impeller_d2 * sizing.impeller_b2 * blockage
