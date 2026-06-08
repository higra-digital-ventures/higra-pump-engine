"""3D inverse blade design solver — Zangeneh prescribed-vorticity method.

Implements a quasi-3D inverse design approach where the blade shape
emerges from a prescribed rVtheta (angular momentum) distribution
across multiple spanwise stations on the meridional plane.

The method works by:
    1. Prescribing rVtheta(m, s) from inlet to outlet at each span station
    2. Computing streamline coordinates on the meridional plane
       via the quasi-normal method
    3. Solving for the blade wrap angle theta(m) at each spanwise
       station using the fundamental inverse design equation:
           dtheta/dm = d(rVtheta)/dm / (cm * (omega*r - Vtheta))
    4. Iterating between the flow field (meridional velocities)
       and blade geometry until the wrap angle converges

References:
    - Zangeneh, M. (1991). A compressible three-dimensional design
      method for radial and mixed flow turbomachinery blades.
      Int. J. Numer. Methods Fluids, 13, 599-624.
    - Zangeneh, M., Goto, A., & Harada, H. (1998). On the design
      criteria for suppression of secondary flows in centrifugal
      and mixed flow impellers. ASME J. Turbomach., 120, 723-735.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums and data models
# ---------------------------------------------------------------------------

class ZangenehLoadingType(str, Enum):
    """Streamwise blade loading distribution type."""

    FRONT = "front"
    MID = "mid"
    AFT = "aft"


@dataclass
class SpanLoadingDef:
    """Prescribed rVtheta loading at one spanwise station.

    Attributes:
        span_fraction: Position from hub (0) to shroud (1).
        nc: Normalized chord position of max loading gradient (LE side).
        nd: Normalized chord position of plateau end (TE side).
        rvt_inlet: rVtheta at leading edge [m^2/s]. Usually 0 (no pre-swirl).
        rvt_outlet: rVtheta at trailing edge [m^2/s]. From Euler equation.
    """

    span_fraction: float = 0.5
    nc: float = 0.20
    nd: float = 0.80
    rvt_inlet: float = 0.0
    rvt_outlet: float = 1.0


@dataclass
class ZangenehSpec:
    """Full specification for the 3D inverse design solver.

    Attributes:
        n_streamlines: Number of spanwise stations (hub to shroud), 3-11.
        loading_type: Default loading shape (front/mid/aft). Overridden
            when per-span loading_defs are provided.
        loading_defs: Per-span loading definitions. If empty, auto-generated
            from loading_type and operating point.
        hub_rz: Hub meridional profile as list of (r, z) points [m].
        shroud_rz: Shroud meridional profile as list of (r, z) points [m].
        flow_rate: Volumetric flow rate Q [m^3/s].
        head: Design head H [m].
        rpm: Rotational speed [rev/min].
        blade_count: Number of blades.
        n_meridional: Number of chordwise discretisation points.
        max_iterations: Maximum solver iterations.
        tolerance: Convergence tolerance on max wrap angle change [rad].
        blockage_factor: Blade blockage factor (0-1).
        rho: Fluid density [kg/m^3].
    """

    n_streamlines: int = 5
    loading_type: ZangenehLoadingType = ZangenehLoadingType.MID
    loading_defs: list[SpanLoadingDef] = field(default_factory=list)

    # Meridional channel geometry
    hub_rz: list[tuple[float, float]] = field(default_factory=list)
    shroud_rz: list[tuple[float, float]] = field(default_factory=list)

    # Operating point
    flow_rate: float = 0.01  # m^3/s
    head: float = 10.0  # m
    rpm: float = 2900.0  # rev/min
    blade_count: int = 7

    # Discretisation
    n_meridional: int = 51

    # Solver parameters
    max_iterations: int = 50
    tolerance: float = 1e-4

    # Physics
    blockage_factor: float = 0.88
    rho: float = 998.0


@dataclass
class ZangenehResult:
    """Output of the Zangeneh 3D inverse design solver.

    All 2-D arrays are indexed [span_index][chord_index].
    """

    # Blade geometry
    blade_angles: list[list[float]]       # beta [deg] at each (span, chord)
    wrap_angles: list[list[float]]         # theta [rad] at each (span, chord)
    camber_lines: list[list[tuple[float, float, float]]]  # (r, theta, z) per span

    # Convergence
    convergence_history: list[float]       # max |delta_theta| per iteration
    converged: bool
    iterations: int

    # Streamline coordinates on the meridional plane
    streamline_coordinates: list[list[tuple[float, float]]]  # (r, z) per span

    # Velocity distributions
    velocity_distributions: list[dict[str, list[float]]]
    # Each dict has: cm, cu, wu, w, beta — at each chord station

    # Derived quantities
    wrap_angle_total: list[float]          # Total wrap angle per span [deg]
    rvt_distributions: list[list[float]]   # rVtheta(m) per span [m^2/s]
    meridional_coords: list[list[float]]   # m(chord) normalised per span


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def zangeneh_inverse_design(spec: ZangenehSpec) -> ZangenehResult:
    """Run the Zangeneh prescribed-vorticity 3D inverse design solver.

    Pipeline:
        1. Build meridional grid: interpolate hub/shroud, distribute
           streamlines using the quasi-normal method.
        2. Generate or validate loading definitions per span.
        3. Iterate:
           a. Compute meridional velocity field (continuity + quasi-normal).
           b. Compute rVtheta(m) at each span from prescribed loading.
           c. Integrate wrap angle theta(m) at each span.
           d. Evaluate blade blockage from current blade shape.
           e. Check convergence on wrap angle change.
        4. Post-process: blade angles, velocity triangles, camber lines.

    Args:
        spec: Complete Zangeneh inverse design specification.

    Returns:
        ZangenehResult with blade geometry and diagnostics.
    """
    omega = 2.0 * math.pi * spec.rpm / 60.0
    g = 9.80665
    n_s = max(3, min(11, spec.n_streamlines))
    n_m = max(11, spec.n_meridional)

    # --- 1. Build meridional grid ------------------------------------------
    hub_rz, shroud_rz = _prepare_meridional_profiles(spec, n_m)
    streamlines = _compute_streamline_grid(hub_rz, shroud_rz, n_s, n_m)
    # streamlines: shape (n_s, n_m, 2) — (r, z) at each (span, chord)

    # Compute meridional arc-length coordinate m(span, chord)
    m_coords = _compute_meridional_coords(streamlines)
    # m_coords: shape (n_s, n_m), normalised to [0, 1]

    # --- 2. Generate loading definitions -----------------------------------
    loading_defs = _resolve_loading_defs(spec, n_s, omega, g)

    # --- 3. Compute rVtheta(m) at each span --------------------------------
    rvt = np.zeros((n_s, n_m), dtype=np.float64)
    for si in range(n_s):
        ld = loading_defs[si]
        rvt[si, :] = _build_rvt_distribution(
            ld, m_coords[si], spec.loading_type,
        )

    # --- 4. Iterative solver -----------------------------------------------
    r_grid = streamlines[:, :, 0]  # (n_s, n_m)
    z_grid = streamlines[:, :, 1]  # (n_s, n_m)

    # Initial meridional velocity from simple continuity
    cm = _compute_meridional_velocity(
        r_grid, z_grid, spec.flow_rate, spec.blockage_factor, spec.blade_count,
    )

    # Wrap angle storage
    theta = np.zeros((n_s, n_m), dtype=np.float64)
    theta_prev = np.zeros_like(theta)

    convergence_history: list[float] = []
    converged = False
    n_iter = 0

    for iteration in range(spec.max_iterations):
        n_iter = iteration + 1
        theta_prev[:] = theta

        # 4a. Integrate wrap angle at each span
        for si in range(n_s):
            theta[si, :] = _integrate_wrap_angle(
                m_coords[si], r_grid[si], cm[si], rvt[si], omega,
            )

        # 4b. Update meridional velocity with blade blockage
        cm = _compute_meridional_velocity_with_blockage(
            r_grid, z_grid, theta, spec.flow_rate,
            spec.blockage_factor, spec.blade_count,
        )

        # 4c. Check convergence
        delta = float(np.max(np.abs(theta - theta_prev)))
        convergence_history.append(delta)

        if delta < spec.tolerance:
            converged = True
            break

    # --- 5. Post-process ---------------------------------------------------
    blade_angles = _compute_blade_angles(
        m_coords, r_grid, cm, rvt, theta, omega,
    )

    velocity_dists = _compute_velocity_distributions(
        r_grid, cm, rvt, omega,
    )

    camber_lines = _build_camber_lines(r_grid, theta, z_grid, n_s, n_m)

    wrap_totals = [
        float(np.degrees(abs(theta[si, -1] - theta[si, 0])))
        for si in range(n_s)
    ]

    streamline_coords = [
        [(float(r_grid[si, mi]), float(z_grid[si, mi])) for mi in range(n_m)]
        for si in range(n_s)
    ]

    return ZangenehResult(
        blade_angles=[row.tolist() for row in blade_angles],
        wrap_angles=[row.tolist() for row in theta],
        camber_lines=camber_lines,
        convergence_history=convergence_history,
        converged=converged,
        iterations=n_iter,
        streamline_coordinates=streamline_coords,
        velocity_distributions=velocity_dists,
        wrap_angle_total=wrap_totals,
        rvt_distributions=[row.tolist() for row in rvt],
        meridional_coords=[row.tolist() for row in m_coords],
    )


# ---------------------------------------------------------------------------
# Quick design: auto-generate spec from operating point
# ---------------------------------------------------------------------------

def zangeneh_quick_design(
    flow_rate: float,
    head: float,
    rpm: float,
    blade_count: int = 7,
    loading_type: ZangenehLoadingType = ZangenehLoadingType.MID,
    n_streamlines: int = 5,
) -> ZangenehResult:
    """Auto-generate a ZangenehSpec and solve.

    Estimates meridional channel geometry from the operating point
    using standard pump design correlations (Gulich, Stepanoff).

    Args:
        flow_rate: Q [m^3/s].
        head: H [m].
        rpm: Rotational speed [rev/min].
        blade_count: Number of blades.
        loading_type: Loading distribution type.
        n_streamlines: Spanwise stations.

    Returns:
        ZangenehResult.
    """
    g = 9.80665
    omega = 2.0 * math.pi * rpm / 60.0

    # Specific speed
    nq = rpm * math.sqrt(flow_rate) / (head ** 0.75)

    # --- Estimate impeller geometry from correlations ----------------------
    # Outlet diameter (Stepanoff)
    ku2 = 1.0 + 0.01 * (nq - 20) if nq > 20 else 1.0
    ku2 = max(0.95, min(1.15, ku2))
    u2 = ku2 * math.sqrt(2.0 * g * head)
    r2 = u2 / omega
    d2 = 2.0 * r2

    # Inlet diameter (based on nq)
    d1_ratio = 0.35 + 0.008 * nq
    d1_ratio = max(0.30, min(0.75, d1_ratio))
    d1 = d2 * d1_ratio
    r1 = d1 / 2.0

    # Hub diameter
    d1_hub = d1 * 0.35

    # Outlet width (Gulich correlation)
    b2_ratio = 0.05 + 0.002 * nq
    b2_ratio = max(0.03, min(0.15, b2_ratio))
    b2 = d2 * b2_ratio

    # Inlet width
    b1 = (d1 - d1_hub) / 2.0

    # --- Build meridional channel ------------------------------------------
    r1_hub = d1_hub / 2.0
    r1_shr = d1 / 2.0
    z_inlet = 0.0
    z_outlet = -b1 * 0.3  # Slight axial shift for mixed-flow element

    # Hub profile: from (r1_hub, z_inlet) curving to (r2, z_outlet)
    hub_rz = _generate_meridional_profile(
        r_start=r1_hub, z_start=z_inlet,
        r_end=r2, z_end=z_outlet - b2 / 2.0,
        n_points=21,
    )

    # Shroud profile
    shroud_rz = _generate_meridional_profile(
        r_start=r1_shr, z_start=z_inlet,
        r_end=r2, z_end=z_outlet + b2 / 2.0,
        n_points=21,
    )

    spec = ZangenehSpec(
        n_streamlines=n_streamlines,
        loading_type=loading_type,
        hub_rz=hub_rz,
        shroud_rz=shroud_rz,
        flow_rate=flow_rate,
        head=head,
        rpm=rpm,
        blade_count=blade_count,
    )

    return zangeneh_inverse_design(spec)


# ---------------------------------------------------------------------------
# Loading templates
# ---------------------------------------------------------------------------

def get_loading_templates() -> list[dict]:
    """Return available loading distribution templates.

    Each template is a dict with name, description, and default
    nc/nd values for hub and shroud.
    """
    return [
        {
            "name": "front",
            "description": (
                "Front-loaded: loading concentrated near the leading edge. "
                "Reduces exit flow non-uniformity. Good for high-Ns pumps."
            ),
            "hub": {"nc": 0.10, "nd": 0.50},
            "shroud": {"nc": 0.15, "nd": 0.55},
        },
        {
            "name": "mid",
            "description": (
                "Mid-loaded: balanced loading distribution. General-purpose "
                "default. Good compromise between efficiency and cavitation."
            ),
            "hub": {"nc": 0.20, "nd": 0.80},
            "shroud": {"nc": 0.20, "nd": 0.80},
        },
        {
            "name": "aft",
            "description": (
                "Aft-loaded: loading concentrated near the trailing edge. "
                "Suppresses secondary flows and jet-wake. Best for low-Ns."
            ),
            "hub": {"nc": 0.40, "nd": 0.90},
            "shroud": {"nc": 0.35, "nd": 0.85},
        },
        {
            "name": "aft_suppressed_secondary",
            "description": (
                "Aft-loaded with hub/shroud differentiation to suppress "
                "secondary flows (Zangeneh 1998). Hub is more aft-loaded "
                "than shroud to control passage vortex."
            ),
            "hub": {"nc": 0.50, "nd": 0.90},
            "shroud": {"nc": 0.25, "nd": 0.75},
        },
        {
            "name": "front_anti_cavitation",
            "description": (
                "Front-loaded at shroud to reduce suction-side velocity "
                "peaks near the leading edge, suppressing cavitation. "
                "Hub remains mid-loaded for efficiency."
            ),
            "hub": {"nc": 0.20, "nd": 0.75},
            "shroud": {"nc": 0.10, "nd": 0.50},
        },
    ]


# ---------------------------------------------------------------------------
# Internal: Meridional grid construction
# ---------------------------------------------------------------------------

def _prepare_meridional_profiles(
    spec: ZangenehSpec,
    n_m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample hub and shroud profiles to uniform n_m points.

    If no profiles are provided in the spec, generates default
    profiles from an estimated geometry.

    Returns:
        (hub_rz, shroud_rz) each shape (n_m, 2).
    """
    if spec.hub_rz and spec.shroud_rz:
        hub_rz = _resample_profile(np.array(spec.hub_rz, dtype=np.float64), n_m)
        shroud_rz = _resample_profile(np.array(spec.shroud_rz, dtype=np.float64), n_m)
    else:
        # Generate default profiles from operating point
        omega = 2.0 * math.pi * spec.rpm / 60.0
        g = 9.80665
        u2 = math.sqrt(2.0 * g * spec.head) * 1.05
        r2 = u2 / omega
        r1 = r2 * 0.45
        r1_hub = r1 * 0.35
        b2 = r2 * 0.10
        b1 = r1 - r1_hub

        hub_pts = _generate_meridional_profile(
            r1_hub, 0.0, r2, -b2 / 2.0, n_m,
        )
        shroud_pts = _generate_meridional_profile(
            r1, 0.0, r2, b2 / 2.0, n_m,
        )
        hub_rz = np.array(hub_pts, dtype=np.float64)
        shroud_rz = np.array(shroud_pts, dtype=np.float64)

    return hub_rz, shroud_rz


def _resample_profile(
    profile: NDArray[np.float64],
    n_target: int,
) -> NDArray[np.float64]:
    """Resample a (N, 2) profile to n_target points with uniform arc length."""
    dr = np.diff(profile[:, 0])
    dz = np.diff(profile[:, 1])
    ds = np.sqrt(dr**2 + dz**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_norm = s / s[-1] if s[-1] > 1e-12 else np.linspace(0, 1, len(s))

    s_new = np.linspace(0.0, 1.0, n_target)
    r_new = np.interp(s_new, s_norm, profile[:, 0])
    z_new = np.interp(s_new, s_norm, profile[:, 1])

    return np.column_stack([r_new, z_new])


def _generate_meridional_profile(
    r_start: float,
    z_start: float,
    r_end: float,
    z_end: float,
    n_points: int,
) -> list[tuple[float, float]]:
    """Generate a smooth meridional profile using a Bezier-like curve.

    Creates a curve from (r_start, z_start) to (r_end, z_end) with
    a smooth transition typical of centrifugal impeller meridional channels.
    """
    t = np.linspace(0.0, 1.0, n_points)

    # Use a cubic Bezier with control points that create a typical
    # centrifugal impeller meridional shape (axial-to-radial transition)
    r_mid1 = r_start + 0.1 * (r_end - r_start)
    z_mid1 = z_start + 0.6 * (z_end - z_start)
    r_mid2 = r_start + 0.5 * (r_end - r_start)
    z_mid2 = z_end

    # Cubic Bezier
    r = (
        (1 - t)**3 * r_start
        + 3 * (1 - t)**2 * t * r_mid1
        + 3 * (1 - t) * t**2 * r_mid2
        + t**3 * r_end
    )
    z = (
        (1 - t)**3 * z_start
        + 3 * (1 - t)**2 * t * z_mid1
        + 3 * (1 - t) * t**2 * z_mid2
        + t**3 * z_end
    )

    return [(float(r[i]), float(z[i])) for i in range(n_points)]


def _compute_streamline_grid(
    hub_rz: NDArray[np.float64],
    shroud_rz: NDArray[np.float64],
    n_s: int,
    n_m: int,
) -> NDArray[np.float64]:
    """Distribute streamlines between hub and shroud using quasi-normal method.

    At each meridional station, the streamlines are distributed such that
    equal flow passes between adjacent streamlines. For the initial
    distribution we use linear interpolation; the solver iteration
    refines this via continuity.

    Returns:
        Array of shape (n_s, n_m, 2) with (r, z) at each grid point.
    """
    grid = np.zeros((n_s, n_m, 2), dtype=np.float64)
    span_fracs = np.linspace(0.0, 1.0, n_s)

    for mi in range(n_m):
        r_hub, z_hub = hub_rz[mi, 0], hub_rz[mi, 1]
        r_shr, z_shr = shroud_rz[mi, 0], shroud_rz[mi, 1]

        for si, sf in enumerate(span_fracs):
            # Area-weighted distribution: bias toward hub for equal dQ
            # For a simple model: use sqrt-weighted interpolation
            # to approximate equal-area rings
            sf_area = sf  # Linear for initial guess
            grid[si, mi, 0] = r_hub + sf_area * (r_shr - r_hub)
            grid[si, mi, 1] = z_hub + sf_area * (z_shr - z_hub)

    return grid


def _compute_meridional_coords(
    streamlines: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute normalised meridional arc-length coordinate m at each point.

    m = 0 at leading edge, m = 1 at trailing edge.

    Returns:
        Array of shape (n_s, n_m) with normalised m values.
    """
    n_s, n_m, _ = streamlines.shape
    m = np.zeros((n_s, n_m), dtype=np.float64)

    for si in range(n_s):
        dr = np.diff(streamlines[si, :, 0])
        dz = np.diff(streamlines[si, :, 1])
        ds = np.sqrt(dr**2 + dz**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]
        m[si, :] = s / s_total if s_total > 1e-12 else np.linspace(0, 1, n_m)

    return m


# ---------------------------------------------------------------------------
# Internal: Loading distribution
# ---------------------------------------------------------------------------

def _resolve_loading_defs(
    spec: ZangenehSpec,
    n_s: int,
    omega: float,
    g: float,
) -> list[SpanLoadingDef]:
    """Resolve per-span loading definitions.

    If the user provided loading_defs, interpolate to the required
    number of spanwise stations. Otherwise, auto-generate from the
    loading_type and operating point.
    """
    if spec.loading_defs:
        # Interpolate user-provided definitions to n_s stations
        return _interpolate_loading_defs(spec.loading_defs, n_s)

    # Auto-generate: compute rvt_outlet from Euler equation
    # H = omega * (rvt_out - rvt_in) / g  =>  rvt_out = g*H/omega
    rvt_outlet = g * spec.head / omega

    # Get template nc/nd for loading type
    templates = {t["name"]: t for t in get_loading_templates()}
    tmpl = templates.get(spec.loading_type.value, templates["mid"])

    span_fracs = np.linspace(0.0, 1.0, n_s)
    defs = []
    for sf in span_fracs:
        # Interpolate nc/nd between hub and shroud template values
        nc = tmpl["hub"]["nc"] + sf * (tmpl["shroud"]["nc"] - tmpl["hub"]["nc"])
        nd = tmpl["hub"]["nd"] + sf * (tmpl["shroud"]["nd"] - tmpl["hub"]["nd"])

        defs.append(SpanLoadingDef(
            span_fraction=float(sf),
            nc=float(nc),
            nd=float(nd),
            rvt_inlet=0.0,  # No pre-swirl
            rvt_outlet=float(rvt_outlet),
        ))

    return defs


def _interpolate_loading_defs(
    defs: list[SpanLoadingDef],
    n_s: int,
) -> list[SpanLoadingDef]:
    """Interpolate user-provided loading defs to n_s spanwise stations."""
    if len(defs) == n_s:
        return defs

    sorted_defs = sorted(defs, key=lambda d: d.span_fraction)
    sf_orig = np.array([d.span_fraction for d in sorted_defs])
    sf_target = np.linspace(0.0, 1.0, n_s)

    nc_vals = np.interp(sf_target, sf_orig, [d.nc for d in sorted_defs])
    nd_vals = np.interp(sf_target, sf_orig, [d.nd for d in sorted_defs])
    rvt_in_vals = np.interp(sf_target, sf_orig, [d.rvt_inlet for d in sorted_defs])
    rvt_out_vals = np.interp(sf_target, sf_orig, [d.rvt_outlet for d in sorted_defs])

    return [
        SpanLoadingDef(
            span_fraction=float(sf_target[i]),
            nc=float(nc_vals[i]),
            nd=float(nd_vals[i]),
            rvt_inlet=float(rvt_in_vals[i]),
            rvt_outlet=float(rvt_out_vals[i]),
        )
        for i in range(n_s)
    ]


def _build_rvt_distribution(
    ld: SpanLoadingDef,
    m: NDArray[np.float64],
    loading_type: ZangenehLoadingType,
) -> NDArray[np.float64]:
    """Build rVtheta(m) distribution at a single span station.

    Uses an S-curve shape controlled by nc (LE transition) and nd
    (TE transition), with the loading_type determining the baseline
    shape within the transition regions.

    The distribution satisfies:
        rVtheta(0) = rvt_inlet
        rVtheta(1) = rvt_outlet
    """
    delta = ld.rvt_outlet - ld.rvt_inlet
    n = len(m)
    rvt = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mi = m[i]
        f = _loading_shape_function(mi, ld.nc, ld.nd, loading_type)
        rvt[i] = ld.rvt_inlet + delta * f

    return rvt


def _loading_shape_function(
    m: float,
    nc: float,
    nd: float,
    loading_type: ZangenehLoadingType,
) -> float:
    """Compute normalised loading shape f(m) in [0, 1].

    Three-region S-curve:
        m < nc:  LE ramp (cubic Hermite)
        nc < m < nd:  central region (linear plateau)
        m > nd:  TE ramp (cubic Hermite)

    The loading_type biases the central region slope.
    """
    nc = max(0.01, min(nc, 0.49))
    nd = max(nc + 0.02, min(nd, 0.99))

    if m <= 0.0:
        return 0.0
    if m >= 1.0:
        return 1.0

    if m <= nc:
        # LE ramp: smooth cubic Hermite from 0 to value at nc
        t = m / nc
        # Fraction of total loading at nc
        f_nc = _central_value_at_nc(nc, nd, loading_type)
        return f_nc * (3.0 * t**2 - 2.0 * t**3)

    if m >= nd:
        # TE ramp: smooth cubic Hermite from value at nd to 1
        t = (m - nd) / (1.0 - nd)
        f_nd = _central_value_at_nd(nc, nd, loading_type)
        return f_nd + (1.0 - f_nd) * (3.0 * t**2 - 2.0 * t**3)

    # Central region: linear interpolation between f(nc) and f(nd)
    f_nc = _central_value_at_nc(nc, nd, loading_type)
    f_nd = _central_value_at_nd(nc, nd, loading_type)
    t = (m - nc) / (nd - nc)
    return f_nc + (f_nd - f_nc) * t


def _central_value_at_nc(
    nc: float, nd: float, loading_type: ZangenehLoadingType,
) -> float:
    """f(nc) value depending on loading type."""
    if loading_type == ZangenehLoadingType.FRONT:
        return 0.55  # More loading completed at nc
    elif loading_type == ZangenehLoadingType.AFT:
        return 0.15  # Less loading completed at nc
    return 0.35  # MID


def _central_value_at_nd(
    nc: float, nd: float, loading_type: ZangenehLoadingType,
) -> float:
    """f(nd) value depending on loading type."""
    if loading_type == ZangenehLoadingType.FRONT:
        return 0.92
    elif loading_type == ZangenehLoadingType.AFT:
        return 0.65
    return 0.80  # MID


# ---------------------------------------------------------------------------
# Internal: Flow field computation
# ---------------------------------------------------------------------------

def _compute_meridional_velocity(
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    flow_rate: float,
    blockage: float,
    blade_count: int,
) -> NDArray[np.float64]:
    """Compute meridional velocity cm from continuity at each grid point.

    At each meridional station, the flow passes through an annular
    area between hub and shroud. The velocity is distributed inversely
    with radius (approximately) to satisfy radial equilibrium.

    Returns:
        cm array of shape (n_s, n_m).
    """
    n_s, n_m = r_grid.shape
    cm = np.zeros((n_s, n_m), dtype=np.float64)

    for mi in range(n_m):
        r_hub = r_grid[0, mi]
        r_shr = r_grid[-1, mi]

        # Channel height at this meridional station
        dr_chan = abs(r_shr - r_hub) if n_s > 1 else r_hub * 0.1
        dz_chan = abs(z_grid[-1, mi] - z_grid[0, mi]) if n_s > 1 else 0.0
        channel_height = math.sqrt(dr_chan**2 + dz_chan**2)
        channel_height = max(channel_height, 1e-6)

        # Mean radius at this station
        r_mean = (r_hub + r_shr) / 2.0

        # Annular area = 2*pi*r_mean * channel_height * blockage
        # Subtract blade thickness blockage
        blade_blockage = 1.0 - blade_count * 0.003 / (2.0 * math.pi * r_mean)
        blade_blockage = max(0.5, min(1.0, blade_blockage))
        area = 2.0 * math.pi * r_mean * channel_height * blockage * blade_blockage

        cm_mean = flow_rate / area if area > 1e-10 else 0.0

        for si in range(n_s):
            # Distribute cm assuming approximately uniform flow
            # (refinement via radial equilibrium would go here)
            r_local = r_grid[si, mi]
            if r_mean > 1e-10 and r_local > 1e-10:
                # Simple 1/r correction for radial equilibrium
                cm[si, mi] = cm_mean * r_mean / r_local
            else:
                cm[si, mi] = cm_mean

    return cm


def _compute_meridional_velocity_with_blockage(
    r_grid: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    theta: NDArray[np.float64],
    flow_rate: float,
    blockage: float,
    blade_count: int,
) -> NDArray[np.float64]:
    """Recompute cm with updated blade blockage from current wrap angles.

    The blade tangential thickness reduces the effective flow area.
    This coupling between blade shape and flow field is what makes
    the solver iterative.
    """
    n_s, n_m = r_grid.shape
    cm = np.zeros((n_s, n_m), dtype=np.float64)

    for mi in range(n_m):
        r_hub = r_grid[0, mi]
        r_shr = r_grid[-1, mi]

        dr_chan = abs(r_shr - r_hub) if n_s > 1 else r_hub * 0.1
        dz_chan = abs(z_grid[-1, mi] - z_grid[0, mi]) if n_s > 1 else 0.0
        channel_height = math.sqrt(dr_chan**2 + dz_chan**2)
        channel_height = max(channel_height, 1e-6)

        r_mean = (r_hub + r_shr) / 2.0

        # Estimate blade angle from theta gradient for blockage
        blade_thickness = 0.003  # m (constant for now)
        if mi > 0 and mi < n_m - 1:
            mid_span = n_s // 2
            dtheta = theta[mid_span, mi + 1] - theta[mid_span, mi - 1]
            dm_val = 2.0  # approximate normalised step
            if abs(dtheta) > 1e-10:
                beta_approx = math.atan2(1.0, r_mean * abs(dtheta / dm_val))
            else:
                beta_approx = math.pi / 2.0
            # Blockage = Z * t / (2*pi*r * sin(beta))
            sin_beta = max(0.1, abs(math.sin(beta_approx)))
            blade_block = blade_count * blade_thickness / (
                2.0 * math.pi * r_mean * sin_beta
            )
        else:
            blade_block = blade_count * blade_thickness / (
                2.0 * math.pi * r_mean
            ) if r_mean > 1e-10 else 0.0

        blade_block = max(0.0, min(0.4, blade_block))
        effective_blockage = blockage * (1.0 - blade_block)

        area = 2.0 * math.pi * r_mean * channel_height * effective_blockage
        cm_mean = flow_rate / area if area > 1e-10 else 0.0

        for si in range(n_s):
            r_local = r_grid[si, mi]
            if r_mean > 1e-10 and r_local > 1e-10:
                cm[si, mi] = cm_mean * r_mean / r_local
            else:
                cm[si, mi] = cm_mean

    return cm


# ---------------------------------------------------------------------------
# Internal: Wrap angle integration
# ---------------------------------------------------------------------------

def _integrate_wrap_angle(
    m: NDArray[np.float64],
    r: NDArray[np.float64],
    cm: NDArray[np.float64],
    rvt: NDArray[np.float64],
    omega: float,
) -> NDArray[np.float64]:
    """Integrate blade wrap angle theta(m) at a single span station.

    The fundamental inverse design equation:

        dtheta/dm = d(rVtheta)/dm / (cm * (omega*r - Vtheta))

    where Vtheta = rVtheta / r.

    This equation states that the blade turning (dtheta/dm) is
    proportional to the prescribed loading rate d(rVtheta)/dm
    and inversely proportional to the relative tangential velocity
    in the rotating frame.

    Uses trapezoidal integration with midpoint averaging for stability.
    """
    n = len(m)
    theta = np.zeros(n, dtype=np.float64)

    # Compute d(rVtheta)/dm
    drvt_dm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            dm = m[1] - m[0] if n > 1 else 1.0
            drvt_dm[i] = (rvt[1] - rvt[0]) / dm if dm > 1e-12 else 0.0
        elif i == n - 1:
            dm = m[-1] - m[-2]
            drvt_dm[i] = (rvt[-1] - rvt[-2]) / dm if dm > 1e-12 else 0.0
        else:
            dm = m[i + 1] - m[i - 1]
            drvt_dm[i] = (rvt[i + 1] - rvt[i - 1]) / dm if dm > 1e-12 else 0.0

    for i in range(1, n):
        dm = m[i] - m[i - 1]
        if dm < 1e-14:
            theta[i] = theta[i - 1]
            continue

        # Trapezoidal: average of values at i-1 and i
        r_avg = 0.5 * (r[i - 1] + r[i])
        cm_avg = 0.5 * (cm[i - 1] + cm[i])
        rvt_avg = 0.5 * (rvt[i - 1] + rvt[i])
        drvt_avg = 0.5 * (drvt_dm[i - 1] + drvt_dm[i])

        # Vtheta = rVtheta / r
        vtheta = rvt_avg / r_avg if r_avg > 1e-10 else 0.0

        # Denominator: cm * (omega*r - Vtheta)
        denom = cm_avg * (omega * r_avg - vtheta)

        if abs(denom) < 1e-10:
            # Near-singular: blade is nearly radial at this point
            theta[i] = theta[i - 1]
            continue

        dtheta = drvt_avg / denom * dm
        theta[i] = theta[i - 1] + dtheta

    return theta


# ---------------------------------------------------------------------------
# Internal: Post-processing
# ---------------------------------------------------------------------------

def _compute_blade_angles(
    m_coords: NDArray[np.float64],
    r_grid: NDArray[np.float64],
    cm: NDArray[np.float64],
    rvt: NDArray[np.float64],
    theta: NDArray[np.float64],
    omega: float,
) -> NDArray[np.float64]:
    """Compute blade angle beta at each (span, chord) point.

    The blade angle in the relative frame is:
        beta = arctan(cm / (omega*r - Vtheta))

    where Vtheta = rVtheta / r.

    Returns:
        Array of shape (n_s, n_m) with blade angles in degrees.
    """
    n_s, n_m = r_grid.shape
    beta = np.zeros((n_s, n_m), dtype=np.float64)

    for si in range(n_s):
        for mi in range(n_m):
            r_val = r_grid[si, mi]
            cm_val = cm[si, mi]
            vtheta = rvt[si, mi] / r_val if r_val > 1e-10 else 0.0
            wu = omega * r_val - vtheta  # Relative tangential velocity

            if abs(wu) < 1e-10:
                beta[si, mi] = 90.0
            else:
                beta[si, mi] = math.degrees(math.atan2(cm_val, wu))

    return beta


def _compute_velocity_distributions(
    r_grid: NDArray[np.float64],
    cm: NDArray[np.float64],
    rvt: NDArray[np.float64],
    omega: float,
) -> list[dict[str, list[float]]]:
    """Compute velocity triangle components at each span.

    Returns a list (one per span) of dicts with keys:
        cm, cu, wu, w, beta
    """
    n_s, n_m = r_grid.shape
    results = []

    for si in range(n_s):
        cm_list: list[float] = []
        cu_list: list[float] = []
        wu_list: list[float] = []
        w_list: list[float] = []
        beta_list: list[float] = []

        for mi in range(n_m):
            r_val = r_grid[si, mi]
            cm_val = float(cm[si, mi])
            cu_val = float(rvt[si, mi] / r_val) if r_val > 1e-10 else 0.0
            u_val = omega * r_val
            wu_val = u_val - cu_val
            w_val = math.sqrt(cm_val**2 + wu_val**2)
            beta_val = math.degrees(math.atan2(cm_val, wu_val)) if wu_val != 0 else 90.0

            cm_list.append(round(cm_val, 4))
            cu_list.append(round(cu_val, 4))
            wu_list.append(round(wu_val, 4))
            w_list.append(round(w_val, 4))
            beta_list.append(round(beta_val, 2))

        results.append({
            "cm": cm_list,
            "cu": cu_list,
            "wu": wu_list,
            "w": w_list,
            "beta": beta_list,
        })

    return results


def _build_camber_lines(
    r_grid: NDArray[np.float64],
    theta: NDArray[np.float64],
    z_grid: NDArray[np.float64],
    n_s: int,
    n_m: int,
) -> list[list[tuple[float, float, float]]]:
    """Build 3D camber lines (r, theta, z) at each spanwise station."""
    camber = []
    for si in range(n_s):
        line = [
            (float(r_grid[si, mi]), float(theta[si, mi]), float(z_grid[si, mi]))
            for mi in range(n_m)
        ]
        camber.append(line)
    return camber
