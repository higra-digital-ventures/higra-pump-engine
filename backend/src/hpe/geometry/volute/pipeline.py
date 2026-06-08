"""Volute sizing pipeline — integrates with 1D sizing results.

Connects the output of hpe.sizing.meanline.run_sizing() to the volute
geometry module, producing a complete VolutePipelineResult with all
key hydraulic and geometric dimensions.

Usage
-----
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.geometry.volute.pipeline import run_volute_pipeline

    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=2900)
    sr = run_sizing(op)
    vr = run_volute_pipeline(sr)
    print(vr.throat_area_m2, vr.exit_diameter_m)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hpe.core.models import SizingResult
from hpe.geometry.volute.models import VoluteParams, VoluteSizing
from hpe.geometry.volute.sizing import size_volute

G = 9.80665  # m/s²


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VolutePipelineResult:
    """Complete output of the volute design pipeline.

    All linear dimensions in metres.
    """

    params: VoluteParams
    sizing: VoluteSizing

    # Derived dimensions
    throat_area_m2: float        # Final volute throat area [m²]
    tongue_radius_m: float       # Cutwater (tongue) radius from shaft centre [m]
    exit_diameter_m: float       # Discharge pipe diameter [m]
    casing_width_m: float        # Maximum casing width (b3) at discharge [m]
    spiral_length_m: float       # Approximate spiral centreline length [m]

    # Validation warnings
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_cu2(sizing_result: SizingResult) -> float:
    """Extract tangential velocity cu2 at impeller outlet.

    Tries the typed VelocityTrianglesResult first; falls back to the legacy
    dict representation for backward compatibility.
    """
    # Typed sub-result (preferred)
    if sizing_result.velocity_triangles_typed is not None:
        return sizing_result.velocity_triangles_typed.outlet.cu

    # Legacy dict path
    vt = sizing_result.velocity_triangles
    outlet = vt.get("outlet", {})
    cu = outlet.get("cu")
    if cu is not None:
        return float(cu)

    # Last-resort: Euler head estimate
    # H_euler = u2 * cu2 / g  →  cu2 = H_euler * g / u2
    u2 = math.pi * sizing_result.impeller_d2 * sizing_result.specific_speed_ns
    # u2 = π * D2 * n / 60
    # We don't have n here directly, so derive from NS definition:
    # Ns = n * sqrt(Q) / H^0.75  but we lack Q — use a conservative fallback
    # Instead, use the slip factor correlation: H = eta_h * u2 * cu2 / g
    # → cu2 ≈ g * H / (eta_h * u2)  with u2 estimated from impeller_d2 and Ns
    raise ValueError(
        "Cannot extract cu2 from SizingResult: velocity_triangles is empty and "
        "velocity_triangles_typed is None. Run sizing with full velocity triangle computation."
    )


def _extract_flow_rate(sizing_result: SizingResult) -> float:
    """Extract flow rate Q [m³/s] from SizingResult.

    Uses velocity triangles to back-calculate from cm2, D2, b2, and blockage,
    or falls back to meridional profile data.
    """
    # Try typed triangles
    if sizing_result.velocity_triangles_typed is not None:
        cm2 = sizing_result.velocity_triangles_typed.outlet.cm
        d2 = sizing_result.impeller_d2
        b2 = sizing_result.impeller_b2
        # Q ≈ π * D2 * b2 * cm2 * blockage_factor
        # Standard blockage ~0.88 for preliminary sizing
        return math.pi * d2 * b2 * cm2 * 0.88

    # Legacy dict
    vt = sizing_result.velocity_triangles
    outlet = vt.get("outlet", {})
    cm2 = outlet.get("cm")
    if cm2 is not None:
        d2 = sizing_result.impeller_d2
        b2 = sizing_result.impeller_b2
        return math.pi * d2 * b2 * float(cm2) * 0.88

    raise ValueError(
        "Cannot extract flow rate from SizingResult: velocity_triangles is empty."
    )


def _spiral_length(r3: float, r_max: float) -> float:
    """Estimate centreline spiral length using Archimedean spiral approximation.

    The spiral grows from radius r3 (at θ=0) to r_max (at θ=2π).
    L ≈ π * (r3 + r_max)  (mean circumference approximation).
    """
    return math.pi * (r3 + r_max)


def _exit_diameter_from_area(area_m2: float) -> float:
    """Diameter of a circular cross-section with the given area."""
    return 2.0 * math.sqrt(area_m2 / math.pi) if area_m2 > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_volute_pipeline(
    sizing_result: SizingResult,
    tongue_clearance: float = 1.05,
    velocity_ratio: float = 0.9,
) -> VolutePipelineResult:
    """Size a spiral volute from a completed 1D sizing result.

    Parameters
    ----------
    sizing_result : SizingResult
        Output of hpe.sizing.meanline.run_sizing().
    tongue_clearance : float
        Ratio of tongue radius to impeller tip radius r2.
        r_tongue = tongue_clearance * r2.  Default 1.05.
    velocity_ratio : float
        Ratio of mean exit velocity to tip speed u2 at discharge.
        Used to estimate the discharge pipe diameter.  Default 0.9.

    Returns
    -------
    VolutePipelineResult
        All volute dimensions plus the intermediate sizing arrays.

    Raises
    ------
    ValueError
        If cu2 or flow rate cannot be extracted from sizing_result.
    """
    warnings: list[str] = []

    r2 = sizing_result.impeller_d2 / 2.0
    b2 = sizing_result.impeller_b2

    # Extract aerodynamic quantities
    cu2 = _extract_cu2(sizing_result)
    q_m3s = _extract_flow_rate(sizing_result)

    # Build VoluteParams
    params = VoluteParams(
        d2=sizing_result.impeller_d2,
        b2=b2,
        flow_rate=q_m3s,
        cu2=cu2,
        radial_gap_ratio=tongue_clearance - 1.0,  # gap = (tongue_clearance - 1) * r2
    )

    # Run area-distribution sizing
    sizing = size_volute(params)

    # Throat (discharge) area
    throat_area_m2 = sizing.discharge_area

    # Tongue radius
    tongue_radius_m = tongue_clearance * r2

    # Exit diameter from discharge area (circular cross-section)
    exit_diameter_m = _exit_diameter_from_area(throat_area_m2)

    # Casing width at discharge — widest cross-section (last station)
    casing_width_m = sizing.widths[-1] if sizing.widths else b2 * 1.5

    # Approximate spiral centreline length
    r_max = max(sizing.radii) if sizing.radii else (r2 + sizing.r3)
    spiral_length_m = _spiral_length(sizing.r3, r_max)

    # Validation warnings
    # 1. Cutwater clearance
    if tongue_clearance < 1.02:
        warnings.append(
            f"Tongue clearance ratio {tongue_clearance:.3f} < 1.02 — "
            "risk of flow recirculation and noise near cutwater."
        )
    if tongue_clearance > 1.15:
        warnings.append(
            f"Tongue clearance ratio {tongue_clearance:.3f} > 1.15 — "
            "large gap may increase recirculation losses."
        )

    # 2. Velocity ratio at discharge
    n_rpm_est = sizing_result.specific_speed_ns  # We don't store n directly in SizingResult
    # Instead check throat velocity against tip speed
    u2 = math.pi * sizing_result.impeller_d2  # placeholder, will warn if area seems off
    if throat_area_m2 > 0:
        v_throat = q_m3s / throat_area_m2
        # u2 computed from impeller_d2 and n_rpm (not available directly):
        # give a relative check instead
        if v_throat > 12.0:
            warnings.append(
                f"Throat velocity {v_throat:.1f} m/s is high — consider a larger discharge area."
            )
        if v_throat < 1.0:
            warnings.append(
                f"Throat velocity {v_throat:.2f} m/s is very low — check flow rate inputs."
            )

    # 3. Exit diameter plausibility
    if exit_diameter_m > 0 and exit_diameter_m < sizing_result.impeller_d2 * 0.25:
        warnings.append(
            "Discharge pipe diameter is less than 25% of impeller D2 — "
            "check flow rate and efficiency inputs."
        )

    # Propagate sizing warnings
    warnings.extend(sizing_result.warnings or [])

    return VolutePipelineResult(
        params=params,
        sizing=sizing,
        throat_area_m2=throat_area_m2,
        tongue_radius_m=tongue_radius_m,
        exit_diameter_m=exit_diameter_m,
        casing_width_m=casing_width_m,
        spiral_length_m=spiral_length_m,
        warnings=warnings,
    )
