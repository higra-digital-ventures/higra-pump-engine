"""Blade thickness distributions for turbomachinery.

Provides 7 streamwise thickness distribution types and spanwise variation
following ADT TurboDesign1 conventions:

1. NACA_4DIGIT  — Standard NACA 00xx symmetric airfoil
2. ELLIPTIC     — Elliptical LE/TE with flat mid-section
3. BIPARABOLIC  — Two parabolas joined at max thickness (common in pumps)
4. LINEAR_TAPER — Linear decrease from LE max to TE min
5. CONSTANT     — Uniform thickness (structural analysis, draft tubes)
6. DCA          — Double circular arc (compressor blades)
7. WEDGE        — Sharp wedge, max at LE to zero at TE (supersonic blades)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class ThicknessType(str, Enum):
    """Supported blade thickness distribution types."""

    NACA_4DIGIT = "naca_4digit"
    ELLIPTIC = "elliptic"
    BIPARABOLIC = "biparabolic"
    LINEAR_TAPER = "linear_taper"
    CONSTANT = "constant"
    DCA = "dca"
    WEDGE = "wedge"


@dataclass
class ThicknessProfile:
    """Blade thickness distribution result."""
    m_normalized: list[float]    # streamwise stations 0→1
    t_normalized: list[float]    # t/t_max at each station
    t_max_over_chord: float      # t_max/chord ratio (e.g., 0.08 for NACA 08xx)
    leading_edge_radius: float   # LE radius / chord (from NACA formula)
    trailing_edge_angle_deg: float  # TE half-wedge angle [deg]
    n_points: int


def naca_thickness(
    t_max_frac: float = 0.08,
    n_points: int = 21,
    close_te: bool = True,
) -> ThicknessProfile:
    """Compute NACA 4-digit symmetric thickness distribution.

    The NACA 4-digit formula (open TE):
        t(x) = 5 * t * (0.2969*√x - 0.1260*x - 0.3516*x² + 0.2843*x³ - 0.1015*x⁴)

    Closed TE modification replaces the last coefficient:
        -0.1015 → -0.1036 (exact closure at x=1)

    Args:
        t_max_frac: Maximum thickness as fraction of chord (e.g., 0.08 = NACA xx08)
        n_points: Number of streamwise stations
        close_te: If True, use closed trailing edge variant

    Returns:
        ThicknessProfile with normalized stations and thickness values
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")

    last_coeff = -0.1036 if close_te else -0.1015

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = []
    t_raw_max = 0.0

    for x in m_stations:
        t = 5.0 * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x ** 2
            + 0.2843 * x ** 3
            + last_coeff * x ** 4
        )
        t_values.append(max(0.0, t))
        if t > t_raw_max:
            t_raw_max = t

    # Normalize to [0, 1]
    t_normalized = [v / t_raw_max if t_raw_max > 0 else 0.0 for v in t_values]

    # LE radius for NACA 4-digit: r_le = 1.1019 * t²
    le_radius = 1.1019 * t_max_frac ** 2

    # TE half-wedge angle
    # At x near 1, dt/dx → determines TE angle
    # For NACA 4-digit, typical TE angle is ~12° for t=0.12
    te_angle_deg = math.degrees(math.atan(2.0 * t_max_frac * 0.234))  # approx

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def ellipse_thickness(
    t_max_frac: float = 0.08,
    le_ratio: float = 2.0,
    te_ratio: float = 1.0,
    n_points: int = 21,
) -> ThicknessProfile:
    """Elliptical leading and trailing edge with flat mid-section.

    Used in ADT TurboDesign1 for the LETE modification. The blade has:
    - Elliptical LE of half-axes (le_ratio * t_max, t_max/2)
    - Flat mid-section at t_max
    - Elliptical TE of half-axes (te_ratio * t_max, t_max/2)

    Args:
        t_max_frac: Maximum thickness as fraction of chord
        le_ratio: LE ellipse semi-major / semi-minor ratio (controls LE sharpness)
        te_ratio: TE ellipse ratio (< 1 = sharp, > 1 = blunt)
        n_points: Number of streamwise stations

    Returns:
        ThicknessProfile
    """
    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = []

    le_extent = le_ratio * t_max_frac   # LE ellipse extends this far in chord
    te_extent = te_ratio * t_max_frac   # TE ellipse extends this far in chord

    for x in m_stations:
        if x <= le_extent and le_extent > 0:
            # LE ellipse: t(x) = t_max * sqrt(1 - ((x - le_extent)/le_extent)²)
            xi = (x - le_extent) / le_extent
            t = t_max_frac * math.sqrt(max(0.0, 1.0 - xi ** 2))
        elif x >= (1.0 - te_extent) and te_extent > 0:
            # TE ellipse
            xi = (x - (1.0 - te_extent)) / te_extent
            t = t_max_frac * math.sqrt(max(0.0, 1.0 - xi ** 2))
        else:
            t = t_max_frac
        t_values.append(t)

    t_raw_max = max(t_values) if t_values else 1.0
    t_normalized = [v / t_raw_max for v in t_values]

    le_radius = t_max_frac / (2 * le_ratio) if le_ratio > 0 else 0.0
    te_angle_deg = math.degrees(math.atan(t_max_frac / te_extent)) if te_extent > 0 else 90.0

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def biparabolic_thickness(
    t_max_frac: float = 0.08,
    n_points: int = 21,
    s_max: float = 0.3,
) -> ThicknessProfile:
    """Biparabolic thickness: two parabolas joined at max thickness location.

    Common in centrifugal pump blades. Front parabola rises from LE to max,
    rear parabola descends from max to TE (thinner tail).

    Args:
        t_max_frac: Maximum thickness as fraction of chord.
        n_points: Number of streamwise stations.
        s_max: Chordwise location of maximum thickness (0-1). Default 0.3.

    Returns:
        ThicknessProfile
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")
    s_max = max(0.05, min(0.95, s_max))

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = []

    for s in m_stations:
        if s <= s_max:
            # Front parabola: rises from 0 at LE to t_max at s_max
            xi = s / s_max
            t = t_max_frac * (2.0 * xi - xi * xi)
        else:
            # Rear parabola: descends from t_max at s_max to 0 at TE
            xi = (s - s_max) / (1.0 - s_max)
            t = t_max_frac * (1.0 - xi * xi)
        t_values.append(max(0.0, t))

    t_raw_max = max(t_values) if t_values else 1.0
    t_normalized = [v / t_raw_max if t_raw_max > 0 else 0.0 for v in t_values]

    le_radius = t_max_frac ** 2 / (2.0 * s_max) if s_max > 0 else 0.0
    te_angle_deg = math.degrees(math.atan(2.0 * t_max_frac / (1.0 - s_max)))

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def linear_taper_thickness(
    t_max_frac: float = 0.08,
    t_min_frac: float = 0.02,
    n_points: int = 21,
) -> ThicknessProfile:
    """Linear taper: thickness decreases linearly from LE to TE.

    Args:
        t_max_frac: Thickness at leading edge (fraction of chord).
        t_min_frac: Thickness at trailing edge (fraction of chord).
        n_points: Number of streamwise stations.

    Returns:
        ThicknessProfile
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")
    t_min_frac = max(0.0, min(t_min_frac, t_max_frac))

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = [t_max_frac + (t_min_frac - t_max_frac) * x for x in m_stations]

    t_raw_max = max(t_values) if t_values else 1.0
    t_normalized = [v / t_raw_max if t_raw_max > 0 else 0.0 for v in t_values]

    le_radius = t_max_frac / 2.0
    te_angle_deg = math.degrees(math.atan(t_max_frac - t_min_frac))

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def constant_thickness(
    t_max_frac: float = 0.08,
    n_points: int = 21,
) -> ThicknessProfile:
    """Constant (uniform) thickness along the entire chord.

    Used for structural analysis, draft tubes, and simplified blade models.

    Args:
        t_max_frac: Uniform thickness as fraction of chord.
        n_points: Number of streamwise stations.

    Returns:
        ThicknessProfile
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_normalized = [1.0] * n_points

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=t_max_frac / 2.0,
        trailing_edge_angle_deg=0.0,
        n_points=n_points,
    )


def dca_thickness(
    t_max_frac: float = 0.08,
    n_points: int = 21,
    s_max: float = 0.5,
) -> ThicknessProfile:
    """Double Circular Arc (DCA) thickness distribution.

    Two circular arcs joined at the maximum thickness location. Common in
    compressor and turbine blades for transonic applications.

    Args:
        t_max_frac: Maximum thickness as fraction of chord.
        n_points: Number of streamwise stations.
        s_max: Chordwise location of maximum thickness (0-1). Default 0.5.

    Returns:
        ThicknessProfile
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")
    s_max = max(0.05, min(0.95, s_max))

    half_t = t_max_frac / 2.0

    # Front arc: circle passing through (0,0) and (s_max, half_t)
    # Radius R_f from geometry: R_f = (s_max^2 + half_t^2) / (2 * half_t)
    r_f = (s_max ** 2 + half_t ** 2) / (2.0 * half_t) if half_t > 0 else 1e6
    # Rear arc: circle passing through (1,0) and (s_max, half_t)
    c_rear = 1.0 - s_max
    r_r = (c_rear ** 2 + half_t ** 2) / (2.0 * half_t) if half_t > 0 else 1e6

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = []

    for x in m_stations:
        if x <= s_max and s_max > 0:
            # Front arc: y = R_f - sqrt(R_f^2 - x^2)  (half-thickness)
            arg = r_f ** 2 - x ** 2
            y_half = r_f - math.sqrt(max(0.0, arg))
            t_values.append(2.0 * y_half)
        else:
            # Rear arc: y = R_r - sqrt(R_r^2 - (1-x)^2)
            dx = 1.0 - x
            arg = r_r ** 2 - dx ** 2
            y_half = r_r - math.sqrt(max(0.0, arg))
            t_values.append(2.0 * y_half)

    t_raw_max = max(t_values) if t_values else 1.0
    t_normalized = [v / t_raw_max if t_raw_max > 0 else 0.0 for v in t_values]

    # LE radius from front arc curvature
    le_radius = 1.0 / (2.0 * r_f) * t_max_frac ** 2 if r_f > 0 else 0.0
    te_angle_deg = math.degrees(math.atan(half_t / c_rear)) if c_rear > 0 else 90.0

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def wedge_thickness(
    t_max_frac: float = 0.08,
    n_points: int = 21,
) -> ThicknessProfile:
    """Wedge thickness: linear from max at LE to zero at TE.

    Used for supersonic blades and sharp-edged profiles.

    Args:
        t_max_frac: Maximum thickness at leading edge as fraction of chord.
        n_points: Number of streamwise stations.

    Returns:
        ThicknessProfile
    """
    if not 0.01 <= t_max_frac <= 0.30:
        raise ValueError(f"t_max_frac must be in [0.01, 0.30], got {t_max_frac}")

    m_stations = [i / (n_points - 1) for i in range(n_points)]
    t_values = [t_max_frac * (1.0 - x) for x in m_stations]

    t_raw_max = max(t_values) if t_values else 1.0
    t_normalized = [v / t_raw_max if t_raw_max > 0 else 0.0 for v in t_values]

    le_radius = t_max_frac / 2.0
    te_angle_deg = math.degrees(math.atan(t_max_frac))

    return ThicknessProfile(
        m_normalized=m_stations,
        t_normalized=t_normalized,
        t_max_over_chord=t_max_frac,
        leading_edge_radius=le_radius,
        trailing_edge_angle_deg=te_angle_deg,
        n_points=n_points,
    )


def get_thickness(
    thickness_type: ThicknessType,
    t_max_frac: float = 0.08,
    n_points: int = 21,
    **kwargs: Any,
) -> ThicknessProfile:
    """Dispatcher: compute thickness distribution for any supported type.

    Args:
        thickness_type: One of ThicknessType enum values.
        t_max_frac: Maximum thickness as fraction of chord.
        n_points: Number of streamwise stations.
        **kwargs: Additional parameters specific to each type:
            - NACA_4DIGIT: close_te (bool)
            - ELLIPTIC: le_ratio (float), te_ratio (float)
            - BIPARABOLIC: s_max (float)
            - LINEAR_TAPER: t_min_frac (float)
            - CONSTANT: (none)
            - DCA: s_max (float)
            - WEDGE: (none)

    Returns:
        ThicknessProfile for the specified distribution type.
    """
    tt = ThicknessType(thickness_type)

    if tt == ThicknessType.NACA_4DIGIT:
        return naca_thickness(
            t_max_frac=t_max_frac,
            n_points=n_points,
            close_te=kwargs.get("close_te", True),
        )
    elif tt == ThicknessType.ELLIPTIC:
        return ellipse_thickness(
            t_max_frac=t_max_frac,
            le_ratio=kwargs.get("le_ratio", 2.0),
            te_ratio=kwargs.get("te_ratio", 1.0),
            n_points=n_points,
        )
    elif tt == ThicknessType.BIPARABOLIC:
        return biparabolic_thickness(
            t_max_frac=t_max_frac,
            n_points=n_points,
            s_max=kwargs.get("s_max", 0.3),
        )
    elif tt == ThicknessType.LINEAR_TAPER:
        return linear_taper_thickness(
            t_max_frac=t_max_frac,
            t_min_frac=kwargs.get("t_min_frac", 0.02),
            n_points=n_points,
        )
    elif tt == ThicknessType.CONSTANT:
        return constant_thickness(
            t_max_frac=t_max_frac,
            n_points=n_points,
        )
    elif tt == ThicknessType.DCA:
        return dca_thickness(
            t_max_frac=t_max_frac,
            n_points=n_points,
            s_max=kwargs.get("s_max", 0.5),
        )
    elif tt == ThicknessType.WEDGE:
        return wedge_thickness(
            t_max_frac=t_max_frac,
            n_points=n_points,
        )
    else:
        raise ValueError(f"Unsupported thickness type: {thickness_type}")


def spanwise_thickness_variation(
    hub_t_max: float,
    mid_t_max: float,
    shr_t_max: float,
    span_pos: float,
) -> float:
    """Interpolate blade thickness at an arbitrary spanwise position.

    Uses quadratic interpolation between hub (0), mid (0.5), shroud (1).

    Args:
        hub_t_max: Max thickness fraction at hub
        mid_t_max: Max thickness fraction at mid-span
        shr_t_max: Max thickness fraction at shroud
        span_pos: Spanwise position 0 (hub) → 1 (shroud)

    Returns:
        Interpolated max thickness fraction
    """
    # Lagrange quadratic through (0, hub), (0.5, mid), (1, shr)
    s = span_pos
    t = (hub_t_max * (s - 0.5) * (s - 1.0) / ((0.0 - 0.5) * (0.0 - 1.0))
         + mid_t_max * s * (s - 1.0) / ((0.5 - 0.0) * (0.5 - 1.0))
         + shr_t_max * s * (s - 0.5) / ((1.0 - 0.0) * (1.0 - 0.5)))
    return max(0.01, t)
