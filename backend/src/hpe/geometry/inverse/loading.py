"""Blade loading distribution for inverse design (rVθ* parameterization).

Defines how rVθ (angular momentum) varies along the meridional
coordinate m ∈ [0, 1] from leading edge to trailing edge.

The loading d(rVθ)/dm determines the blade shape: fore-loaded
distributions produce blades with more curvature near the inlet,
while aft-loaded distributions shift curvature toward the outlet.

This module also implements the S-curve loading shape used in
TURBOdesign1. The loading distribution rVθ*(m) controls the work
distribution along the blade chord from LE (m=0) to TE (m=1).

References:
    - Zangeneh, M. (1991). A compressible three-dimensional design
      method for radial and mixed flow turbomachinery blades.
    - Goto, A. & Zangeneh, M. (2002). Hydrodynamic design of pump
      diffuser using inverse design method and CFD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpe.geometry.inverse.models import BladeLoadingSpec, LoadingType


# ---------------------------------------------------------------------------
# S-curve loading distribution (TURBOdesign1 style) — B1
# ---------------------------------------------------------------------------

@dataclass
class LoadingDistribution:
    """S-curve rVθ* distribution parameters for one span (hub or shroud)."""

    nc: float = 0.20        # Position of max loading gradient (LE side)
    nd: float = 0.80        # Position of plateau end (TE side)
    slope: float = 1.5      # TE slope (positive = forward loaded)
    drvt_le: float = 0.0    # LE derivative (0 = no LE loading spike)
    rvt_te: float = 0.523   # Target rVθ* at TE [m²/s] (from Euler equation)
    n_points: int = 51      # Number of chord points


@dataclass
class BladeLoadingResult:
    """Result of blade loading calculation."""

    m_norm: list[float]       # Normalized chord position [0..1]
    rvt: list[float]          # rVθ* distribution [m²/s]
    drvt_dm: list[float]      # d(rVθ*)/dm — loading rate
    ps_excess: list[float]    # Pressure-side velocity excess (normalized)
    ss_excess: list[float]    # Suction-side velocity excess (normalized)


def calc_loading_distribution(
    hub: LoadingDistribution,
    shroud: LoadingDistribution,
    n_span: int = 3,
) -> dict:
    """Calculate rVθ* distribution from hub to shroud.

    Args:
        hub: Loading parameters at hub.
        shroud: Loading parameters at shroud.
        n_span: Number of spanwise positions (currently returns hub/mid/shroud).

    Returns:
        Dict with 'hub', 'mid', 'shroud' BladeLoadingResult objects.
    """
    results = {}
    spans = {"hub": 0.0, "mid": 0.5, "shroud": 1.0}

    for span_name, xi in spans.items():
        # Interpolate parameters linearly between hub and shroud
        nc = hub.nc + xi * (shroud.nc - hub.nc)
        nd = hub.nd + xi * (shroud.nd - hub.nd)
        slope = hub.slope + xi * (shroud.slope - hub.slope)
        drvt_le = hub.drvt_le + xi * (shroud.drvt_le - hub.drvt_le)
        rvt_te = hub.rvt_te + xi * (shroud.rvt_te - hub.rvt_te)

        n_pts = hub.n_points
        m = [i / (n_pts - 1) for i in range(n_pts)]
        rvt = _s_curve(m, nc, nd, slope, drvt_le, rvt_te)
        drvt = _gradient(rvt, m)

        # Pressure/suction side excess velocities (simplified linear model).
        # delta_w proportional to d(rVθ*)/dm (loading rate).
        max_drvt = max(abs(d) for d in drvt) if drvt else 1.0
        ps_excess = [-0.5 * d / max_drvt if max_drvt > 0 else 0.0 for d in drvt]
        ss_excess = [+0.5 * d / max_drvt if max_drvt > 0 else 0.0 for d in drvt]

        results[span_name] = BladeLoadingResult(
            m_norm=m,
            rvt=rvt,
            drvt_dm=drvt,
            ps_excess=ps_excess,
            ss_excess=ss_excess,
        )

    return results


def calc_blade_pressure_distribution(
    loading: BladeLoadingResult,
    w_inlet: float,
    w_outlet: float,
    rho: float = 998.0,
) -> dict:
    """Calculate PS and SS velocity/pressure distributions from loading.

    Args:
        loading: BladeLoadingResult with loading distribution.
        w_inlet: Inlet relative velocity [m/s].
        w_outlet: Outlet relative velocity [m/s].
        rho: Fluid density [kg/m³].

    Returns:
        Dict with m_norm, w_ps, w_ss, cp_ps, cp_ss (pressure coefficients).
    """
    n = len(loading.m_norm)
    w_ref = (w_inlet + w_outlet) / 2.0

    w_ps: list[float] = []
    w_ss: list[float] = []
    cp_ps: list[float] = []
    cp_ss: list[float] = []

    for i in range(n):
        t = loading.m_norm[i]
        # Mean relative velocity (linear interpolation)
        w_mean = w_inlet + t * (w_outlet - w_inlet)
        # PS/SS deviation from loading
        delta_w = loading.drvt_dm[i] * 0.1  # scale factor

        wps = max(0.0, w_mean - abs(delta_w))
        wss = w_mean + abs(delta_w)

        w_ps.append(round(wps, 3))
        w_ss.append(round(wss, 3))

        # Pressure coefficient Cp = (p - p_ref) / (0.5 * rho * w_ref²)
        if w_ref > 1e-6:
            cp_ps.append(round(1.0 - (wps / w_ref) ** 2, 4))
            cp_ss.append(round(1.0 - (wss / w_ref) ** 2, 4))
        else:
            cp_ps.append(0.0)
            cp_ss.append(0.0)

    return {
        "m_norm": loading.m_norm,
        "w_ps": w_ps,
        "w_ss": w_ss,
        "cp_ps": cp_ps,
        "cp_ss": cp_ss,
        "w_ref": round(w_ref, 3),
    }


def _s_curve(
    m: list[float],
    nc: float,
    nd: float,
    slope: float,
    drvt_le: float,
    rvt_te: float,
) -> list[float]:
    """Generate S-curve rVθ* from LE to TE.

    The curve rises from 0 at LE (with optional LE spike controlled by
    drvt_le), has a plateau between NC and ND, and reaches rvt_te at TE
    with the given slope.
    """
    result = []
    for mi in m:
        if mi <= nc:
            # LE region: quadratic rise with LE derivative control
            t = mi / nc if nc > 0 else 0.0
            le_spike = drvt_le * mi * (1 - mi / nc) if nc > 0 else 0.0
            val = rvt_te * (0.5 * t ** 2 * (3 - 2 * t)) + le_spike
        elif mi <= nd:
            # Plateau region
            t = (mi - nc) / (nd - nc) if (nd - nc) > 0 else 1.0
            val = rvt_te * (nc / (nc + (1 - nd))) * (1 + t * (nd - nc) / nd)
            val = min(val, rvt_te * 0.95)
        else:
            # TE region: approach rvt_te with given slope
            t = (mi - nd) / (1 - nd) if (1 - nd) > 0 else 1.0
            val_plateau = rvt_te * (nc / (nc + (1 - nd))) * (1 + (nd - nc) / nd)
            val_plateau = min(val_plateau, rvt_te * 0.95)
            val = val_plateau + (rvt_te - val_plateau) * (3 * t ** 2 - 2 * t ** 3)
            # Apply slope modification at TE
            val += slope * 0.01 * (t - 1) * (1 - nd)
        result.append(max(0.0, val))

    # Normalize so TE = rvt_te
    if result and result[-1] > 1e-9:
        scale = rvt_te / result[-1]
        result = [v * scale for v in result]

    return result


def _gradient(y: list[float], x: list[float]) -> list[float]:
    """Compute numerical gradient dy/dx using central differences."""
    n = len(y)
    grad = [0.0] * n
    for i in range(n):
        if i == 0:
            grad[i] = (
                (y[1] - y[0]) / (x[1] - x[0])
                if n > 1 and (x[1] - x[0]) > 0
                else 0.0
            )
        elif i == n - 1:
            grad[i] = (
                (y[-1] - y[-2]) / (x[-1] - x[-2])
                if (x[-1] - x[-2]) > 0
                else 0.0
            )
        else:
            dx = x[i + 1] - x[i - 1]
            grad[i] = (y[i + 1] - y[i - 1]) / dx if dx > 0 else 0.0
    return grad


def compute_rvt_distribution(
    spec: BladeLoadingSpec,
    m_coords: list[float],
) -> list[float]:
    """Compute rVθ distribution along the meridional coordinate.

    Given the loading specification and a set of meridional coordinates
    m ∈ [0, 1], returns rVθ at each station.

    The distribution satisfies:
        rVθ(0) = rvt_inlet
        rVθ(1) = rvt_outlet
        The shape between is controlled by loading_type.

    Args:
        spec: Blade loading specification.
        m_coords: Normalized meridional coordinates (0=LE, 1=TE).

    Returns:
        List of rVθ values [m²/s] at each m coordinate.
    """
    rvt_in = spec.rvt_inlet
    rvt_out = spec.rvt_outlet
    delta_rvt = rvt_out - rvt_in

    if spec.loading_type == LoadingType.CUSTOM and spec.loading_control_points:
        return _custom_loading(spec, m_coords)

    rvt_values: list[float] = []
    for m in m_coords:
        f = _loading_shape(m, spec.loading_type)
        rvt_values.append(rvt_in + delta_rvt * f)

    return rvt_values


def compute_loading_derivative(
    rvt_values: list[float],
    m_coords: list[float],
) -> list[float]:
    """Compute d(rVθ)/dm — the blade loading derivative.

    This is the actual "loading" that drives blade curvature.
    High values of |d(rVθ)/dm| correspond to high blade turning.

    Args:
        rvt_values: rVθ at each meridional station.
        m_coords: Normalized meridional coordinates.

    Returns:
        d(rVθ)/dm at each station (central differences, forward/backward at ends).
    """
    n = len(rvt_values)
    drvt_dm: list[float] = []

    for i in range(n):
        if i == 0:
            # Forward difference
            dm = m_coords[1] - m_coords[0]
            deriv = (rvt_values[1] - rvt_values[0]) / dm if dm > 0 else 0.0
        elif i == n - 1:
            # Backward difference
            dm = m_coords[-1] - m_coords[-2]
            deriv = (rvt_values[-1] - rvt_values[-2]) / dm if dm > 0 else 0.0
        else:
            # Central difference
            dm = m_coords[i + 1] - m_coords[i - 1]
            deriv = (rvt_values[i + 1] - rvt_values[i - 1]) / dm if dm > 0 else 0.0

        drvt_dm.append(deriv)

    return drvt_dm


def compute_spanwise_rvt(
    spec: BladeLoadingSpec,
    span_fractions: list[float],
) -> list[float]:
    """Compute rVθ_outlet at each spanwise station.

    For free-vortex design: rVθ = constant across span.
    For constant-rVt: same as free-vortex (rVθ = const by definition).
    For custom: interpolate from user-specified values.

    Args:
        spec: Blade loading specification.
        span_fractions: Spanwise positions (0=hub, 1=shroud).

    Returns:
        rVθ_outlet at each span station [m²/s].
    """
    from hpe.geometry.inverse.models import StackingCondition

    if spec.stacking == StackingCondition.CUSTOM and spec.spanwise_rvt:
        return _interpolate_spanwise(spec.spanwise_rvt, span_fractions)

    # Free vortex: rVθ = constant across span
    return [spec.rvt_outlet] * len(span_fractions)


def _loading_shape(m: float, loading_type: LoadingType) -> float:
    """Compute the normalized loading shape function f(m) ∈ [0, 1].

    f(0) = 0, f(1) = 1. The derivative df/dm controls where the
    blade loading is concentrated.

    Fore-loaded:  steep rise near LE → gentle near TE
    Aft-loaded:   gentle near LE → steep rise near TE
    Mid-loaded:   symmetric S-curve (sinusoidal)
    """
    if loading_type == LoadingType.FORE_LOADED:
        # Power law: f = 1 - (1-m)^2.5
        return 1.0 - (1.0 - m) ** 2.5

    elif loading_type == LoadingType.AFT_LOADED:
        # Power law: f = m^2.5
        return m**2.5

    elif loading_type == LoadingType.MID_LOADED:
        # Sinusoidal: f = 0.5 * (1 - cos(pi*m))
        return 0.5 * (1.0 - math.cos(math.pi * m))

    return m  # Linear fallback


def _custom_loading(
    spec: BladeLoadingSpec,
    m_coords: list[float],
) -> list[float]:
    """Compute rVθ from custom control points using monotone interpolation.

    Control points define (m, weight) pairs where weight ∈ [0, 1]
    indicates the cumulative fraction of ΔrVθ applied at that m.
    Endpoints (0, 0) and (1, 1) are always included.
    """
    rvt_in = spec.rvt_inlet
    delta_rvt = spec.rvt_outlet - spec.rvt_inlet

    # Build sorted control points with endpoints
    pts = [(0.0, 0.0)] + sorted(spec.loading_control_points) + [(1.0, 1.0)]

    rvt_values: list[float] = []
    for m in m_coords:
        # Find bounding control points
        f = _piecewise_linear_interp(pts, m)
        rvt_values.append(rvt_in + delta_rvt * f)

    return rvt_values


def _piecewise_linear_interp(
    pts: list[tuple[float, float]],
    x: float,
) -> float:
    """Piecewise linear interpolation."""
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.0
            return y0 + t * (y1 - y0)

    return pts[-1][1]


def _interpolate_spanwise(
    spanwise_rvt: list[tuple[float, float]],
    span_fractions: list[float],
) -> list[float]:
    """Interpolate custom spanwise rVθ distribution."""
    pts = sorted(spanwise_rvt)
    return [_piecewise_linear_interp(pts, s) for s in span_fractions]
