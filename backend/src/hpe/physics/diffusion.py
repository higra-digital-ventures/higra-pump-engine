"""Lieblein diffusion ratio and de Haller criterion for blade passages."""

from __future__ import annotations


def calc_diffusion_ratio(w1: float, w2: float) -> float:
    """Lieblein diffusion ratio D = w2/w1.

    D > 0.7 → risk of boundary layer separation (Gulich §4.2).
    D > 0.75 → stall likely.

    Args:
        w1: Inlet relative velocity [m/s].
        w2: Outlet relative velocity [m/s].

    Returns:
        Diffusion ratio w2/w1 (de Haller number).
    """
    if w1 < 1e-6:
        return 1.0
    return w2 / w1


def calc_de_haller(w1: float, w2: float) -> float:
    """De Haller number (same as diffusion ratio for pumps)."""
    return calc_diffusion_ratio(w1, w2)


def lieblein_diffusion_factor(
    w1: float, w2: float,
    delta_cu: float, chord: float, pitch: float,
) -> float:
    """Lieblein Diffusion Factor (compressor cascade, extended for pumps).

    DF = 1 - w2/w1 + delta_cu/(2*w1*pitch/chord)

    DF > 0.6 → stall for pumps.
    """
    solidity = chord / pitch  # σ
    if w1 < 1e-6:
        return 0.0
    df = 1.0 - w2 / w1 + delta_cu / (2.0 * w1 * (1.0 / solidity if solidity > 0 else 1.0))
    return df


def check_diffusion_warnings(w1: float, w2: float) -> list[str]:
    """Return list of diffusion-related warnings."""
    warnings = []
    dh = calc_diffusion_ratio(w1, w2)
    if dh < 0.6:
        warnings.append(
            f"De Haller number {dh:.2f} < 0.6: high diffusion, separation risk."
        )
    elif dh < 0.7:
        warnings.append(
            f"De Haller number {dh:.2f} in caution zone (0.60–0.70)."
        )
    return warnings
