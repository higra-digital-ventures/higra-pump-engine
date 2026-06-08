"""Physics-based validators for geometry and performance data.

Rule-based checks that catch obvious errors before they enter
the ML pipeline. These are deterministic (not statistical).
"""

from __future__ import annotations

from dataclasses import dataclass

from hpe.core.models import PerformanceMetrics, SizingResult


@dataclass
class ValidationResult:
    """Result of validation check."""

    valid: bool
    errors: list[str]  # Critical issues
    warnings: list[str]  # Non-critical concerns


def validate_geometry(sizing: SizingResult) -> ValidationResult:
    """Validate that sizing geometry is physically consistent.

    Args:
        sizing: SizingResult to validate.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Dimensional checks
    if sizing.impeller_d2 <= 0:
        errors.append("D2 must be positive")
    if sizing.impeller_d1 <= 0:
        errors.append("D1 must be positive")
    if sizing.impeller_d1 >= sizing.impeller_d2:
        errors.append(f"D1 ({sizing.impeller_d1:.4f}) must be less than D2 ({sizing.impeller_d2:.4f})")
    if sizing.impeller_b2 <= 0:
        errors.append("b2 must be positive")

    # Blade count
    if not 3 <= sizing.blade_count <= 15:
        errors.append(f"Blade count {sizing.blade_count} outside valid range [3, 15]")

    # Blade angles
    if not 5 < sizing.beta1 < 70:
        warnings.append(f"Inlet blade angle beta1={sizing.beta1:.1f} deg outside typical range [5, 70]")
    if not 10 < sizing.beta2 < 50:
        warnings.append(f"Outlet blade angle beta2={sizing.beta2:.1f} deg outside typical range [10, 50]")

    # Tip speed
    u2 = sizing.velocity_triangles.get("outlet", {}).get("u", 0)
    if u2 > 60:
        warnings.append(f"Tip speed u2={u2:.1f} m/s exceeds 60 m/s — risk of erosion and noise")
    if u2 > 80:
        errors.append(f"Tip speed u2={u2:.1f} m/s exceeds 80 m/s — structural limit")

    # Specific speed
    if sizing.specific_speed_nq < 5:
        warnings.append(f"Very low Nq={sizing.specific_speed_nq:.0f} — consider positive displacement pump")
    if sizing.specific_speed_nq > 300:
        warnings.append(f"Very high Nq={sizing.specific_speed_nq:.0f} — consider axial machine")

    # D1/D2 ratio
    ratio = sizing.impeller_d1 / sizing.impeller_d2 if sizing.impeller_d2 > 0 else 0
    if not 0.2 < ratio < 0.9:
        warnings.append(f"D1/D2 ratio = {ratio:.2f} outside typical range [0.2, 0.9]")

    # Euler head consistency
    h_euler = sizing.velocity_triangles.get("euler_head", 0)
    if h_euler <= 0:
        errors.append("Euler head must be positive")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_performance(perf: PerformanceMetrics) -> ValidationResult:
    """Validate that performance metrics are physically plausible.

    Args:
        perf: PerformanceMetrics to validate.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Efficiency ranges
    for name, val in [
        ("hydraulic", perf.hydraulic_efficiency),
        ("volumetric", perf.volumetric_efficiency),
        ("mechanical", perf.mechanical_efficiency),
        ("total", perf.total_efficiency),
    ]:
        if not 0 <= val <= 1:
            errors.append(f"{name} efficiency = {val:.3f} outside [0, 1]")
        elif val > 0.98:
            warnings.append(f"{name} efficiency = {val:.1%} suspiciously high")

    # Head
    if perf.head < 0:
        errors.append(f"Negative head ({perf.head:.1f} m)")

    # Power
    if perf.power < 0:
        errors.append(f"Negative power ({perf.power:.1f} W)")

    # NPSH
    if perf.npsh_required < 0:
        errors.append(f"Negative NPSHr ({perf.npsh_required:.1f} m)")

    # Consistency: eta_total ~ eta_h * eta_v * eta_m
    expected_total = perf.hydraulic_efficiency * perf.volumetric_efficiency * perf.mechanical_efficiency
    if abs(expected_total - perf.total_efficiency) > 0.05:
        warnings.append(
            f"Total efficiency ({perf.total_efficiency:.3f}) inconsistent with "
            f"product of components ({expected_total:.3f})"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
