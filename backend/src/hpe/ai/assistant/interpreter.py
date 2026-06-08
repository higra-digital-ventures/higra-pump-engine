"""Engineering result interpreter — translates numeric results to clear text.

Transforms SizingResult and PerformanceMetrics into human-readable
analysis that engineers can use to understand and communicate design decisions.
"""

from __future__ import annotations

from hpe.core.models import PerformanceMetrics, SizingResult


def interpret_sizing(sizing: SizingResult) -> str:
    """Generate a human-readable interpretation of sizing results.

    Args:
        sizing: SizingResult from the sizing module.

    Returns:
        Multi-paragraph text interpretation.
    """
    nq = sizing.specific_speed_nq
    imp_type = sizing.meridional_profile.get("impeller_type", "unknown")
    d2_mm = sizing.impeller_d2 * 1000
    eta = sizing.estimated_efficiency
    power_kw = sizing.estimated_power / 1000
    npsh = sizing.estimated_npsh_r

    lines: list[str] = []

    # Classification
    if nq < 25:
        lines.append(f"This is a low specific speed design (Nq={nq:.0f}), typical of high-head, low-flow pumps.")
        lines.append("Low-Nq impellers have narrow passages and are prone to disk friction losses.")
    elif nq < 70:
        lines.append(f"This is a standard radial impeller (Nq={nq:.0f}), the most common type for industrial centrifugal pumps.")
        lines.append("Good balance between efficiency and operating range.")
    elif nq < 160:
        lines.append(f"This is a mixed-flow design (Nq={nq:.0f}), combining radial and axial flow characteristics.")
        lines.append("Suitable for high-flow, moderate-head applications.")
    else:
        lines.append(f"This is an axial-flow design (Nq={nq:.0f}), optimized for very high flow rates with low head.")

    # Dimensions
    lines.append(f"\nImpeller outer diameter: {d2_mm:.0f} mm with {sizing.blade_count} blades.")
    lines.append(f"Blade angles: {sizing.beta1:.1f} deg (inlet) to {sizing.beta2:.1f} deg (outlet).")

    # Efficiency assessment
    if eta > 0.85:
        lines.append(f"\nEstimated efficiency of {eta:.1%} is excellent for this specific speed range.")
    elif eta > 0.75:
        lines.append(f"\nEstimated efficiency of {eta:.1%} is good, within expected range for Nq={nq:.0f}.")
    elif eta > 0.65:
        lines.append(f"\nEstimated efficiency of {eta:.1%} is moderate. Optimization may yield 3-5pp improvement.")
    else:
        lines.append(f"\nEstimated efficiency of {eta:.1%} is below typical. Review operating point selection.")

    # Power
    lines.append(f"Required shaft power: {power_kw:.1f} kW.")

    # Cavitation
    if npsh > 8:
        lines.append(f"\nNPSH required of {npsh:.1f} m is high. Ensure adequate suction conditions or consider booster pump.")
    elif npsh > 5:
        lines.append(f"\nNPSH required of {npsh:.1f} m is moderate. Verify suction system can provide sufficient margin.")
    else:
        lines.append(f"\nNPSH required of {npsh:.1f} m is low, favorable for installation flexibility.")

    # Warnings
    if sizing.warnings:
        lines.append("\nAttention points:")
        for w in sizing.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def interpret_performance(
    perf: PerformanceMetrics,
    design_head: float,
) -> str:
    """Interpret performance metrics at a specific operating point.

    Args:
        perf: PerformanceMetrics from physics evaluation.
        design_head: Target head [m] for comparison.

    Returns:
        Interpretation text.
    """
    lines: list[str] = []

    # Head comparison
    head_ratio = perf.head / design_head if design_head > 0 else 0
    if 0.95 <= head_ratio <= 1.05:
        lines.append(f"Head of {perf.head:.1f} m matches design target ({design_head:.1f} m) well.")
    elif head_ratio > 1.05:
        lines.append(f"Head of {perf.head:.1f} m exceeds design target ({design_head:.1f} m) by {(head_ratio-1)*100:.0f}%.")
        lines.append("Excess head can be throttled but reduces system efficiency.")
    else:
        lines.append(f"Head of {perf.head:.1f} m is {(1-head_ratio)*100:.0f}% below design target ({design_head:.1f} m).")
        lines.append("Impeller may need upsizing or speed increase.")

    # Efficiency breakdown
    lines.append("\nEfficiency breakdown:")
    lines.append(f"  Hydraulic: {perf.hydraulic_efficiency:.1%} (flow losses in impeller)")
    lines.append(f"  Volumetric: {perf.volumetric_efficiency:.1%} (leakage through clearances)")
    lines.append(f"  Mechanical: {perf.mechanical_efficiency:.1%} (disk friction + bearings)")
    lines.append(f"  Total: {perf.total_efficiency:.1%}")

    # Identify dominant loss
    losses = {
        "hydraulic": 1 - perf.hydraulic_efficiency,
        "volumetric": 1 - perf.volumetric_efficiency,
        "mechanical": 1 - perf.mechanical_efficiency,
    }
    dominant = max(losses, key=losses.get)  # type: ignore[arg-type]
    lines.append(f"\nDominant loss source: {dominant} ({losses[dominant]:.1%} of total).")

    return "\n".join(lines)
