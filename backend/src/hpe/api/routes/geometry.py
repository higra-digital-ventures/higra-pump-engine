"""Geometry API routes — 3D visualization and CAD export.

Geometry improvements:
    #9  — target_wrap_angle: iterative beta2 adjustment to hit desired wrap.
    #10 — blade_profile: "logarithmic" (default) | "bezier" (Bézier camber).
    #11 — le_radius / te_radius: non-zero LE/TE thickness via smooth distribution.
    #12 — t_hub / t_shroud: spanwise thickness variation (root-to-tip taper).
    #14 — lean_angle / sweep_angle: stacking law for blade lean and sweep.
"""

from __future__ import annotations

import math
import tempfile
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["geometry"])


class GeometryRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    n_blade_points: int = Field(60, ge=20, le=200)
    n_span_points: int = Field(16, ge=4, le=64)
    resolution_preset: Optional[str] = Field(
        None,
        description=(
            "Resolution preset: 'low' (30x8), 'medium' (60x16), "
            "'high' (120x32), 'ultra' (200x64). "
            "Overrides n_blade_points and n_span_points when set."
        ),
    )
    # Advanced geometry parameters
    target_wrap_angle: Optional[float] = Field(None, description="Target blade wrap angle [deg] (#9). Typical 100-160.")
    blade_profile: str = Field("logarithmic", description="Blade camber profile: 'logarithmic' | 'bezier' (#10)")
    le_radius: Optional[float] = Field(None, description="Leading edge radius [m] (#11). Default = blade_thickness/4")
    te_radius: Optional[float] = Field(None, description="Trailing edge radius [m] (#11). Default = blade_thickness/6")
    t_hub: Optional[float] = Field(None, description="Blade thickness at hub [m] (#12). Default = blade_thickness")
    t_shroud: Optional[float] = Field(None, description="Blade thickness at shroud [m] (#12). Default = 0.6*blade_thickness")
    lean_angle: float = Field(0.0, description="Blade lean angle [deg] (#14). Tangential stacking offset hub→shroud")
    sweep_angle: float = Field(0.0, description="Blade sweep angle [deg] (#14). Axial stacking offset hub→shroud")
    add_splitters: bool = Field(False, description="Add splitter blades at half pitch")
    splitter_start: float = Field(0.40, ge=0.25, le=0.65, description="Meridional start fraction for splitters")
    machine_type: str = Field("centrifugal_pump", description="Machine type: 'centrifugal_pump' | 'francis_turbine'")


class BladePoint3D(BaseModel):
    x: float
    y: float
    z: float


class BladeSurface(BaseModel):
    ps: List[List[BladePoint3D]]
    ss: List[List[BladePoint3D]]
    ps_pressure: List[List[float]] = []   # normalized pressure 0→1 per vertex
    ss_pressure: List[List[float]] = []   # normalized pressure 0→1 per vertex


class ImpellerGeometry(BaseModel):
    blade_surfaces: List[BladeSurface]
    splitter_surfaces: List[BladeSurface] = []  # half-length blades interleaved
    splitter_count: int = 0
    splitter_start_fraction: float = 0.0
    hub_profile: List[BladePoint3D]
    shroud_profile: List[BladePoint3D]
    blade_count: int
    d2: float
    d1: float
    b2: float
    actual_wrap_angle: float = 0.0   # (#9) computed wrap angle of blade 0


# ---------------------------------------------------------------------------
# Meridional curves
# ---------------------------------------------------------------------------

def _meridional_curves(
    r1: float, r1_hub: float, r2: float,
    b1: float, b2: float, n_chord: int,
    machine_type: str = "centrifugal_pump",
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Meridional curves for impeller/runner.

    Coordinate system: r = radial, z = axial (z=0 is the back-disc plane).
    The impeller extends from z=0 (back disc) upward to z=z_eye (inlet).

    Hub profile (inner wall):
      - Back disc: flat at z=0 from r=r_shaft to r=r_bend
      - Bend: quarter-circle arc from radial to axial
      - Eye tube: nearly vertical up to z=z_eye at r=r1_hub

    Shroud profile (outer wall):
      - Offset from hub by passage width b(t)
      - At inlet: offset radially outward (r direction)
      - At outlet: offset axially (z direction)

    For Francis turbines:
      - Longer axial extent (z_inlet = r1 * 1.5)
      - Shorter radial zone (t < 0.45), longer bend (0.45 to 0.75)
      - Deeper bowl-shaped hub
    """
    # Blade channel: from r2 (outlet) to r1 (inlet eye radius)
    # Blades do NOT extend below r1 — the shaft/eye tube is visual only
    z_base = b2 * 0.10          # hub surface height above back-disc

    if machine_type == "francis_turbine":
        z_inlet = r1 * 1.5     # deeper axial extent for Francis
        arc_r = z_inlet         # arc radius = z_inlet
        radial_end = 0.45       # shorter radial zone
        bend_end = 0.75         # longer bend zone
    else:
        z_inlet = r1 * 0.85    # inlet height at r1 (centrifugal pump)
        arc_r = z_inlet         # arc radius determines how deep the eye goes
        radial_end = 0.65       # standard radial zone
        bend_end = 1.0          # bend fills remainder

    hub_rz: list[tuple[float, float]] = []
    shroud_rz: list[tuple[float, float]] = []

    for i in range(n_chord):
        t = i / (n_chord - 1)
        # t=0: outlet (r=r2, z≈0), t=1: inlet (r=r1, z=z_inlet)

        # --- HUB ---
        if t < radial_end:
            # Radial zone: flat disc from r2 toward r1+arc_r
            s = t / radial_end
            r_h = r2 - (r2 - r1 - arc_r) * s
            z_h = z_base
        elif t < bend_end:
            # Bend: quarter-arc sweeping from radial to axial
            s = (t - radial_end) / (bend_end - radial_end)
            arc = (math.pi / 2) * (s ** 2.0)
            r_h = r1 + arc_r * (1.0 - math.sin(arc))
            z_h = z_base + arc_r * (1.0 - math.cos(arc))
        else:
            # Axial extension (Francis only — after bend)
            s = (t - bend_end) / (1.0 - bend_end) if bend_end < 1.0 else 1.0
            r_h = r1 + arc_r * (1.0 - 1.0)  # stays at r1
            z_h = z_base + arc_r + s * (z_inlet - arc_r) * 0.3

        # Passage width: b2 at outlet → b1 at inlet
        b_t = b2 + t * (b1 - b2)

        # --- SHROUD ---
        # Linear r from r2 to r1, z offset tapers
        r_s = r2 + t * (r1 - r2)
        z_s = z_h + b_t * (1.0 - t * 0.5)

        hub_rz.append((r_h, z_h))
        shroud_rz.append((r_s, z_s))

    return hub_rz, shroud_rz


def _hub_with_shaft(
    hub_rz: list[tuple[float, float]],
    r1_hub: float,
) -> list[tuple[float, float]]:
    """Add shaft stub, nose cone and hub back-disc for visual completeness.

    The hub_rz goes from outlet (r2, z=0) to inlet (r1_hub, z_eye).
    We add:
    - Back disc with rim: at z=0, from r_shaft to r2 (closes the back)
    - Shaft stub at inlet with a smooth conical nose
    """
    if not hub_rz:
        return hub_rz

    # hub_rz[0] = outlet (r2, z_base), hub_rz[-1] = inlet (r1, z_inlet)
    r_out, z_out = hub_rz[0]   # outlet (r=r2)
    r_in, z_in = hub_rz[-1]    # inlet (r=r1)

    # Bore radius for the disc (visible hole)
    r_bore = r_out * 0.18       # bore = 18% of D2/2 — matches real pump bore

    # Revolution profile:
    # 1. Bore inner edge at back → disc outer at back (rim)
    # 2. Disc outer front face
    # 3. Hub blade channel (r2 → r1)
    # Profile stops at r1 — eye is open
    profile = [
        (r_bore, -0.003),       # bore edge, back face
        (r_out, -0.003),        # disc outer, back face (3mm rim)
        (r_out, 0.0),           # disc outer, front face
    ]
    profile += hub_rz           # blade channel hub (r2→r1)
    # Eye is OPEN — no wall closing it
    return profile


# ---------------------------------------------------------------------------
# Thickness distribution (#11)
# ---------------------------------------------------------------------------

def _naca_thickness_at(t_norm: float, t_max: float, le_r: float, te_r: float) -> float:
    """NACA 4-digit symmetric thickness at normalized chord station t (0→1).

    Formula (closed TE variant):
        y(t) = 5*t_max*(0.2969*√t - 0.1260*t - 0.3516*t² + 0.2843*t³ - 0.1036*t⁴)

    Blended with LE/TE rounding via le_r and te_r minimum thickness floor.
    """
    if t_max <= 0:
        return 0.0
    t = max(1e-8, t_norm)
    # NACA 4-digit
    y = 5.0 * (0.2969 * t**0.5 - 0.1260 * t - 0.3516 * t**2 + 0.2843 * t**3 - 0.1036 * t**4)
    naca_thick = t_max * max(0.0, y)
    # Floor: ensure LE/TE have minimum rounding thickness
    floor = le_r + (te_r - le_r) * t_norm
    return max(floor, naca_thick)


# ---------------------------------------------------------------------------
# Camber line integration
# ---------------------------------------------------------------------------

def _integrate_camber_logarithmic(
    hub_rz: list[tuple[float, float]],
    shroud_rz: list[tuple[float, float]],
    beta1_rad: float, beta2_rad: float,
    s: float,  # span fraction 0=hub 1=shroud
    machine_type: str = "centrifugal_pump",
) -> list[tuple[float, float, float]]:
    """Build camber (r, z, theta) by logarithmic spiral integration.

    hub_rz goes from outlet (r2, z=0) to inlet (r1, z=z_eye).
    Integration is done REVERSED (inlet→outlet) for correct wrap direction,
    then the result is flipped back to match the outlet→inlet order.
    """
    n = len(hub_rz)

    # Build reversed arrays: inlet (i=0) → outlet (i=n-1)
    hub_rev = list(reversed(hub_rz))
    shr_rev = list(reversed(shroud_rz))

    # Target wrap: scale beta to achieve target wrap
    r_outlet = hub_rev[-1][0] + s * (shr_rev[-1][0] - hub_rev[-1][0])
    r_inlet = hub_rev[0][0] + s * (shr_rev[0][0] - hub_rev[0][0])
    # Target wrap at this span (hub gets more, shroud gets less)
    if machine_type == "francis_turbine":
        target_wrap = math.radians(50 + 15 * (1 - s))  # hub~65°, shroud~50°
    else:
        target_wrap = math.radians(80 + 20 * (1 - s))  # hub~100°, shroud~80°

    # Compute what beta_eff gives target wrap: wrap = ln(r2/r1)/tan(beta_eff)
    r_ratio = r_outlet / max(r_inlet, 0.001)
    if r_ratio > 1 and target_wrap > 0:
        beta_eff = math.atan(math.log(r_ratio) / target_wrap)
    else:
        beta_eff = (beta1_rad + beta2_rad) / 2

    theta = 0.0
    camber_rev = []
    for i in range(n):
        t = i / (n - 1)
        r = hub_rev[i][0] + s * (shr_rev[i][0] - hub_rev[i][0])
        z = hub_rev[i][1] + s * (shr_rev[i][1] - hub_rev[i][1])
        # Blend between sizing betas and effective beta for wrap control
        beta = beta1_rad + t * (beta2_rad - beta1_rad)
        beta_use = 0.3 * beta + 0.7 * beta_eff  # mostly use effective beta
        camber_rev.append((r, z, theta))
        if i < n - 1:
            r_next = hub_rev[i+1][0] + s * (shr_rev[i+1][0] - hub_rev[i+1][0])
            dr = r_next - r
            r_mid = (r + r_next) / 2
            if abs(math.tan(beta_use)) > 1e-10 and r_mid > 1e-6 and abs(dr) > 1e-8:
                theta += dr / (r_mid * math.tan(beta_use))

    # Reverse back to outlet→inlet order (matching hub_rz)
    return list(reversed(camber_rev))


def _integrate_camber_bezier(
    hub_rz: list[tuple[float, float]],
    shroud_rz: list[tuple[float, float]],
    beta1_rad: float, beta2_rad: float,
    s: float,
    machine_type: str = "centrifugal_pump",
) -> list[tuple[float, float, float]]:
    """Bézier camber line (#10): cubic Bézier wrap angle distribution.

    The wrap angle theta(r) follows a cubic Bézier curve from (r_in, 0) to (r_out, theta_total),
    where theta_total is computed from the logarithmic solution and used as the endpoint.
    The two interior control points are set at 1/3 and 2/3 of the r-range,
    with angles slightly biased toward the leading edge for a more aggressive
    wrap at inlet — typical for high-efficiency pump blades.
    """
    # First, compute the total wrap using logarithmic to anchor the Bezier endpoint
    log_camber = _integrate_camber_logarithmic(hub_rz, shroud_rz, beta1_rad, beta2_rad, s, machine_type=machine_type)
    theta_total = log_camber[-1][2]
    n = len(hub_rz)

    # Bézier control points in normalized (t, theta/theta_total) space
    # P0=(0,0), P1=(0.3, 0.15), P2=(0.7, 0.7), P3=(1,1) — S-curve
    def bezier_theta(t_norm: float) -> float:
        p0, p1, p2, p3 = 0.0, 0.15, 0.70, 1.0
        b = (1-t_norm)**3*p0 + 3*(1-t_norm)**2*t_norm*p1 + 3*(1-t_norm)*t_norm**2*p2 + t_norm**3*p3
        return b * theta_total

    camber = []
    for i in range(n):
        t = i / (n - 1)
        r = hub_rz[i][0] + s * (shroud_rz[i][0] - hub_rz[i][0])
        z = hub_rz[i][1] + s * (shroud_rz[i][1] - hub_rz[i][1])
        theta = bezier_theta(t)
        camber.append((r, z, theta))
    return camber


# ---------------------------------------------------------------------------
# Surface generation
# ---------------------------------------------------------------------------

def _build_blade_surface(
    hub_rz: list[tuple[float, float]],
    shroud_rz: list[tuple[float, float]],
    beta1_rad: float, beta2_rad: float,
    blade_thickness: float,
    angular_offset: float,
    n_span: int,
    blade_profile: str = "logarithmic",
    le_radius: float = 0.0,
    te_radius: float = 0.0,
    t_hub: float | None = None,
    t_shroud: float | None = None,
    lean_rad: float = 0.0,    # lean stacking [rad]
    sweep_m: float = 0.0,     # sweep stacking [m]
    machine_type: str = "centrifugal_pump",
) -> BladeSurface:
    t_hub_v = t_hub if t_hub is not None else blade_thickness
    t_shroud_v = t_shroud if t_shroud is not None else blade_thickness  # uniform (ADT style)

    ps_surface: list[list[BladePoint3D]] = []
    ss_surface: list[list[BladePoint3D]] = []

    for k in range(n_span):
        s = k / (n_span - 1)

        # Spanwise thickness (#12)
        t_max = t_hub_v + s * (t_shroud_v - t_hub_v)

        # Choose camber integration method (#10)
        if blade_profile == "bezier":
            camber = _integrate_camber_bezier(hub_rz, shroud_rz, beta1_rad, beta2_rad, s, machine_type=machine_type)
        else:
            camber = _integrate_camber_logarithmic(hub_rz, shroud_rz, beta1_rad, beta2_rad, s, machine_type=machine_type)

        # Stacking offsets (#14)
        lean_offset = lean_rad * s
        n_chord = len(camber)

        ps_row, ss_row = [], []
        for i, (r, z, theta_c) in enumerate(camber):
            t_norm = i / (n_chord - 1)

            # Thickness at this chord station (#11)
            thick = _naca_thickness_at(t_norm, t_max, le_radius, te_radius)

            # Tangential offset: d_theta = half_thickness / r
            d_theta = thick / (2.0 * r) if r > 1e-10 else 0.0

            # Lean + sweep stacking (#14)
            z_offset = sweep_m * s
            angle = theta_c + angular_offset + lean_offset

            # PS: rotate backward by d_theta
            ps_row.append(BladePoint3D(
                x=r * math.cos(angle - d_theta) * 1000,
                y=r * math.sin(angle - d_theta) * 1000,
                z=(z + z_offset) * 1000,
            ))
            # SS: rotate forward by d_theta
            ss_row.append(BladePoint3D(
                x=r * math.cos(angle + d_theta) * 1000,
                y=r * math.sin(angle + d_theta) * 1000,
                z=(z + z_offset) * 1000,
            ))

        ps_surface.append(ps_row)
        ss_surface.append(ss_row)

    # Simple pressure distribution model:
    # PS (pressure side): pressure decreases from LE to throat, then recovers
    # SS (suction side): pressure drops sharply at LE, minimum at throat
    n_chord = len(ps_surface[0]) if ps_surface else 0
    ps_pressure: list[list[float]] = []
    ss_pressure: list[list[float]] = []
    for k in range(n_span):
        ps_row_p: list[float] = []
        ss_row_p: list[float] = []
        for i in range(n_chord):
            t = i / max(1, n_chord - 1)
            # PS: high at LE, dips at mid (0.6), recovers at TE
            ps_p = 0.7 + 0.3 * (1 - 4 * (t - 0.3) ** 2) if 0 < t < 0.6 else 0.5 + 0.4 * t
            ps_p = max(0.1, min(1.0, ps_p))
            # SS: lowest pressure near throat (t~0.3), builds up toward TE
            ss_p = 0.15 + 0.7 * t + 0.15 * t ** 2
            ss_p = max(0.05, min(0.95, ss_p))
            ps_row_p.append(round(ps_p, 3))
            ss_row_p.append(round(ss_p, 3))
        ps_pressure.append(ps_row_p)
        ss_pressure.append(ss_row_p)

    return BladeSurface(ps=ps_surface, ss=ss_surface, ps_pressure=ps_pressure, ss_pressure=ss_pressure)


def _compute_wrap_angle(
    hub_rz: list[tuple[float, float]],
    shroud_rz: list[tuple[float, float]],
    beta1_rad: float, beta2_rad: float,
    machine_type: str = "centrifugal_pump",
) -> float:
    """Compute wrap angle [deg] at mid-span for camber diagnostics (#9).

    After the inlet-to-outlet integration + reversal, the maximum theta
    may sit at either end of the camber list (outlet or inlet).  We take
    the absolute maximum across all stations so the result is never 0.
    """
    camber = _integrate_camber_logarithmic(hub_rz, shroud_rz, beta1_rad, beta2_rad, 0.5, machine_type=machine_type)
    return math.degrees(max(abs(pt[2]) for pt in camber))


def _adjust_beta2_for_wrap(
    hub_rz, shroud_rz, beta1_rad: float, beta2_rad: float,
    target_wrap_rad: float, max_iter: int = 12,
    machine_type: str = "centrifugal_pump",
) -> float:
    """Bisection search on beta2 to achieve target wrap angle (#9)."""
    lo, hi = math.radians(15.0), math.radians(45.0)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        w = math.radians(_compute_wrap_angle(hub_rz, shroud_rz, beta1_rad, mid, machine_type=machine_type))
        if abs(w - target_wrap_rad) < math.radians(0.5):
            return mid
        # Higher beta2 → smaller wrap angle (more radial blade)
        if w > target_wrap_rad:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _revolution_profile(rz_curve: list[tuple[float, float]]) -> list[BladePoint3D]:
    return [BladePoint3D(x=r * 1000, y=0.0, z=z * 1000) for r, z in rz_curve]


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post("/geometry/impeller", response_model=ImpellerGeometry)
def get_impeller_geometry(req: GeometryRequest) -> ImpellerGeometry:
    """Generate 3D blade surfaces with advanced geometry parameters."""
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    mp = sizing.meridional_profile
    d1 = sizing.impeller_d1
    d2 = sizing.impeller_d2
    b2 = sizing.impeller_b2
    b1 = float(mp.get("b1", b2 * 1.2))
    r1 = d1 / 2.0
    r1_hub = float(mp.get("d1_hub", d1 * 0.35)) / 2.0
    r2 = d2 / 2.0

    # Fix 1: Match ADT thickness — 1.3% of D2 as ACTUAL max thickness
    # NACA formula gives peak at ~50% of t_max, so t_max = 2 * desired_actual
    blade_thickness = max(0.004, min(0.016, d2 * 0.026))

    beta1_rad = math.radians(sizing.beta1)
    beta2_rad = math.radians(sizing.beta2)

    # Resolution presets override explicit n_blade_points / n_span_points
    # [Action 5] Resolutions — 'high' now matches ADT chordwise (89 pts)
    _RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
        "low": (30, 8),
        "medium": (60, 12),
        "high": (89, 20),
        "ultra": (150, 40),
    }
    if req.resolution_preset and req.resolution_preset in _RESOLUTION_PRESETS:
        n_chord, n_span = _RESOLUTION_PRESETS[req.resolution_preset]
    else:
        n_chord = req.n_blade_points
        n_span = req.n_span_points

    machine_type = req.machine_type

    hub_rz, shroud_rz = _meridional_curves(r1, r1_hub, r2, b1, b2, n_chord, machine_type=machine_type)
    hub_rz_visual = _hub_with_shaft(hub_rz, r1_hub)

    # #9 — wrap angle targeting
    if req.target_wrap_angle is not None:
        target_rad = math.radians(req.target_wrap_angle)
        beta2_rad = _adjust_beta2_for_wrap(hub_rz, shroud_rz, beta1_rad, beta2_rad, target_rad, machine_type=machine_type)

    # Fix 4: LE 25% / TE 10% (ADT-like sharp trailing edge)
    le_r = req.le_radius if req.le_radius is not None else blade_thickness * 0.25
    te_r = req.te_radius if req.te_radius is not None else blade_thickness * 0.10
    # Fix 2: No spanwise tapering — uniform thickness like ADT
    t_hub = req.t_hub if req.t_hub is not None else blade_thickness
    t_shroud = req.t_shroud if req.t_shroud is not None else blade_thickness

    # Stacking (#14)
    lean_rad = math.radians(req.lean_angle)
    sweep_m = math.tan(math.radians(req.sweep_angle)) * b2 if req.sweep_angle != 0 else 0.0

    pitch = 2.0 * math.pi / sizing.blade_count
    blade_surfaces: list[BladeSurface] = []
    for b in range(sizing.blade_count):
        surf = _build_blade_surface(
            hub_rz, shroud_rz,
            beta1_rad, beta2_rad,
            blade_thickness,
            angular_offset=b * pitch,
            n_span=n_span,
            blade_profile=req.blade_profile,
            le_radius=le_r,
            te_radius=te_r,
            t_hub=t_hub,
            t_shroud=t_shroud,
            lean_rad=lean_rad,
            sweep_m=sweep_m,
            machine_type=machine_type,
        )
        blade_surfaces.append(surf)

    actual_wrap = _compute_wrap_angle(hub_rz, shroud_rz, beta1_rad, beta2_rad, machine_type=machine_type)

    splitter_surfaces: list[BladeSurface] = []
    splitter_start_frac = req.splitter_start

    if req.add_splitters:
        half_pitch = math.pi / sizing.blade_count  # half pitch offset
        # Splitter starts at splitter_start_frac of meridional chord
        start_idx = max(0, int(splitter_start_frac * n_chord))

        # Build truncated meridional for splitter (starts at start_idx)
        hub_rz_split = hub_rz[start_idx:]
        shroud_rz_split = shroud_rz[start_idx:]

        # Adjust beta1_rad for splitter (use midpoint beta since it starts mid-chord)
        t_start = start_idx / (n_chord - 1)
        beta_split_rad = beta1_rad + t_start * (beta2_rad - beta1_rad)

        # Slightly thinner blades
        split_thickness = blade_thickness * 0.85
        split_le = le_r * 0.8
        split_te = te_r

        for b in range(sizing.blade_count):
            ang_off = b * (2.0 * math.pi / sizing.blade_count) + half_pitch
            surf = _build_blade_surface(
                hub_rz_split, shroud_rz_split,
                beta_split_rad, beta2_rad,
                split_thickness,
                angular_offset=ang_off,
                n_span=n_span,
                blade_profile=req.blade_profile,
                le_radius=split_le,
                te_radius=split_te,
                t_hub=req.t_hub,
                t_shroud=req.t_shroud,
                lean_rad=lean_rad,
                sweep_m=sweep_m,
                machine_type=machine_type,
            )
            splitter_surfaces.append(surf)

    return ImpellerGeometry(
        blade_surfaces=blade_surfaces,
        splitter_surfaces=splitter_surfaces,
        splitter_count=len(splitter_surfaces),
        splitter_start_fraction=req.splitter_start if req.add_splitters else 0.0,
        hub_profile=_revolution_profile(hub_rz_visual),
        shroud_profile=_revolution_profile(shroud_rz),
        blade_count=sizing.blade_count,
        d2=d2, d1=d1, b2=b2,
        actual_wrap_angle=actual_wrap,
    )


# ---------------------------------------------------------------------------
# Blade quality endpoint (B9)
# ---------------------------------------------------------------------------

@router.get("/geometry/quality")
def get_geometry_quality(
    flow_rate: float,
    head: float,
    rpm: float,
    wrap_hub: float = 0.0,
    wrap_shr: float = 0.0,
) -> dict:
    """Return blade geometric quality metrics for a design point (B9).

    Runs 1D sizing to obtain impeller geometry, then evaluates blade
    quality indicators: wrap variation, minimum passage, curvature
    coefficient, LE bow, sweep, lean angles, and a composite score.

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        rpm: Rotational speed [RPM].
        wrap_hub: Wrap angle at hub [deg] (0 = auto-estimated from geometry).
        wrap_shr: Wrap angle at shroud [deg] (0 = auto-estimated).

    Returns:
        BladeQualityMetrics fields plus impeller geometry context.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.geometry.runner.quality import calc_blade_quality

    op = OperatingPoint(flow_rate=flow_rate, head=head, rpm=rpm)
    sizing = run_sizing(op)

    metrics = calc_blade_quality(
        d2=sizing.impeller_d2,
        d1=sizing.impeller_d1,
        b2=sizing.impeller_b2,
        blade_count=sizing.blade_count,
        beta2=sizing.beta2,
        beta1=sizing.beta1,
        wrap_hub=wrap_hub,
        wrap_shr=wrap_shr,
    )

    return {
        "wrap_angle_variation_deg": metrics.wrap_angle_variation,
        "channel_min_distance_m": metrics.channel_min_distance,
        "curvature_variation_coeff": metrics.curvature_variation_coeff,
        "le_bow_ratio": metrics.le_bow_ratio,
        "le_sweep_deg": metrics.le_sweep_deg,
        "max_lean_deg": metrics.max_lean_deg,
        "avg_lean_deg": metrics.avg_lean_deg,
        "quality_score": metrics.quality_score,
        "warnings": metrics.warnings,
        # Context
        "d2_m": sizing.impeller_d2,
        "d1_m": sizing.impeller_d1,
        "b2_m": sizing.impeller_b2,
        "blade_count": sizing.blade_count,
        "beta1_deg": sizing.beta1,
        "beta2_deg": sizing.beta2,
    }


# ---------------------------------------------------------------------------
# Meridional Bezier endpoint (A3)
# ---------------------------------------------------------------------------

class BezierControlPointInput(BaseModel):
    r: float = Field(..., description="Radial coordinate [m]")
    z: float = Field(..., description="Axial coordinate [m]")


class BezierMeridionalRequest(BaseModel):
    hub_cp: List[BezierControlPointInput] = Field(
        ..., min_length=4, max_length=4,
        description="4 Bézier control points for the hub curve (r,z) [m]",
    )
    shroud_cp: List[BezierControlPointInput] = Field(
        ..., min_length=4, max_length=4,
        description="4 Bézier control points for the shroud curve (r,z) [m]",
    )
    le_curvature_coeff: float = Field(
        0.05, ge=0.0, le=0.30,
        description="Leading-edge curvature coefficient (ADT LE CURVATURE COEFF). Range 0–0.30.",
    )
    te_inclination: float = Field(
        0.0, ge=-45.0, le=45.0,
        description="Trailing-edge inclination angle [deg]. Positive = more radial TE.",
    )
    n_points: int = Field(50, ge=10, le=200, description="Number of discretisation points per curve.")


class BezierMeridionalResponse(BaseModel):
    hub_curve: List[dict]     # list of {"r": float, "z": float}
    shroud_curve: List[dict]


@router.post("/geometry/meridional/bezier", response_model=BezierMeridionalResponse)
def meridional_bezier_endpoint(req: BezierMeridionalRequest) -> BezierMeridionalResponse:
    """Generate hub and shroud meridional curves from 4 Bézier control points each.

    Supports leading-edge curvature coefficient and trailing-edge inclination,
    matching ADT TURBOdesign1 ADVANCED IMPELLER MERIDIONAL parameterisation.
    """
    from hpe.geometry.runner.meridional_advanced import BezierMeridional

    hub_cp = [(p.r, p.z) for p in req.hub_cp]
    shroud_cp = [(p.r, p.z) for p in req.shroud_cp]

    bm = BezierMeridional(
        hub_cp=hub_cp,
        shroud_cp=shroud_cp,
        le_curvature_coeff=req.le_curvature_coeff,
        te_inclination=req.te_inclination,
        n_points=req.n_points,
    )

    hub_pts = [{"r": r, "z": z} for r, z in bm.hub_curve()]
    shroud_pts = [{"r": r, "z": z} for r, z in bm.shroud_curve()]

    return BezierMeridionalResponse(hub_curve=hub_pts, shroud_curve=shroud_pts)


# ---------------------------------------------------------------------------
# Volute 2D flow solver endpoint (D1–D5)
# ---------------------------------------------------------------------------

class VolSolveRequest(BaseModel):
    """Request body for the volute flow solver."""
    # VoluteGeometry fields
    r2: float = Field(..., gt=0, description="Impeller outlet radius [m]")
    b2: float = Field(..., gt=0, description="Impeller outlet width [m]")
    r3: float = Field(..., gt=0, description="Cutwater (tongue) radius [m]")
    section_type: str = Field("SEMICIRCLE", description="Cross-section type: SEMICIRCLE | ELLIPSE | RECTANGLE | TRAPEZOID")
    volute_type: str = Field("single_radial", description="Volute type: single_radial | single_tangential | double | semi_double | asymmetric_ext | double_entry | axial_entry")
    semi_major: float = Field(0.0, ge=0, description="Ellipse major axis [m]")
    semi_minor: float = Field(0.0, ge=0, description="Ellipse minor axis [m]")
    rect_width: float = Field(0.0, ge=0, description="Rectangle width [m]")
    rect_height: float = Field(0.0, ge=0, description="Rectangle height [m]")
    tongue_radius: float = Field(0.002, gt=0, description="Tongue fillet radius [m]")
    tube_length: float = Field(0.2, gt=0, description="Discharge tube length [m]")
    tube_diameter: float = Field(0.08, gt=0, description="Discharge tube diameter [m]")
    tube_angle_deg: float = Field(0.0, description="Discharge tube angle [deg]")
    rho: float = Field(998.0, gt=0, description="Fluid density [kg/m³]")
    # Operating point
    flow_rate: float = Field(..., gt=0, description="Flow rate Q [m³/s]")
    head: float = Field(..., gt=0, description="Pump head H [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")


@router.post("/geometry/volute/solve")
def solve_volute_endpoint(req: VolSolveRequest) -> dict:
    """Solve volute 2D flow and compute performance (Gülich 2014 §7.5).

    Returns total head loss, static pressure recovery coefficient,
    throat area, angular section velocities, and loss coefficient.
    Supports SEMICIRCLE, ELLIPSE, RECTANGLE, and TRAPEZOID cross-sections,
    and all seven volute types (single/double/tangential/axial-entry, etc.).
    """
    from hpe.physics.volute_solver import (
        CrossSectionType, VoluteGeometry, VoluteType, solve_volute,
    )

    try:
        section_type = CrossSectionType(req.section_type.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown section_type '{req.section_type}'. "
                   f"Choose from: {[e.value for e in CrossSectionType]}",
        )

    try:
        volute_type = VoluteType(req.volute_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown volute_type '{req.volute_type}'. "
                   f"Choose from: {[e.value for e in VoluteType]}",
        )

    geom = VoluteGeometry(
        r2=req.r2,
        b2=req.b2,
        r3=req.r3,
        section_type=section_type,
        volute_type=volute_type,
        semi_major=req.semi_major,
        semi_minor=req.semi_minor,
        rect_width=req.rect_width,
        rect_height=req.rect_height,
        tongue_radius=req.tongue_radius,
        tube_length=req.tube_length,
        tube_diameter=req.tube_diameter,
        tube_angle_deg=req.tube_angle_deg,
        rho=req.rho,
    )

    result = solve_volute(volute=geom, flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)

    return {
        "total_head_loss_m": result.total_head_loss_m,
        "static_pressure_recovery": result.static_pressure_recovery,
        "throat_area_m2": result.throat_area_m2,
        "scroll_exit_area_m2": result.scroll_exit_area_m2,
        "mean_velocity_ms": result.mean_velocity_ms,
        "discharge_velocity_ms": result.discharge_velocity_ms,
        "loss_coefficient": result.loss_coefficient,
        "sections": result.sections,
    }


# ---------------------------------------------------------------------------
# G1 — Structured .geo export endpoint
# ---------------------------------------------------------------------------

@router.get("/geometry/export/geo")
def export_geo_endpoint(flow_rate: float, head: float, rpm: float, unit: str = "m") -> dict:
    """Export blade in structured .geo format (BladeGen / TurboGrid).

    Runs 1D sizing to derive impeller geometry, then produces (X, R, theta)
    coordinate tables for both pressure and suction surfaces at n_span=5
    spanwise positions and n_chord=21 chordwise stations.

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        rpm: Rotational speed [RPM].
        unit: Output unit for coordinates, "m" or "mm".

    Returns:
        Dict with 'ps', 'ss' (.geo file contents), 'format', 'n_span', 'n_chord'.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.geometry.runner.export import export_geo_both_surfaces

    op = OperatingPoint(flow_rate=flow_rate, head=head, rpm=rpm)
    s = run_sizing(op)
    return export_geo_both_surfaces(
        d2=s.impeller_d2,
        d1=s.impeller_d1,
        b2=s.impeller_b2,
        beta1=s.beta1,
        beta2=s.beta2,
        blade_count=s.blade_count,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# G2 — BladeGen .inf + .curve export endpoint
# ---------------------------------------------------------------------------

@router.get("/geometry/export/bladegen")
def export_bladegen_endpoint(flow_rate: float, head: float, rpm: float) -> dict:
    """Export blade in ANSYS BladeGen .inf + .curve format.

    Runs 1D sizing to derive impeller geometry, then produces:
    - An .inf file with machine type, rotation axis, and span definitions.
    - A .curve file with (m', theta_deg) blade wrap coordinates per span.

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        rpm: Rotational speed [RPM].

    Returns:
        Dict with 'inf' (.inf file content), 'curve' (.curve file content),
        and 'format' = "bladegen".
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.geometry.runner.export import export_bladegen_curve, export_bladegen_inf

    op = OperatingPoint(flow_rate=flow_rate, head=head, rpm=rpm)
    s = run_sizing(op)
    return {
        "inf": export_bladegen_inf(
            d2=s.impeller_d2,
            d1=s.impeller_d1,
            b2=s.impeller_b2,
            beta1=s.beta1,
            beta2=s.beta2,
            blade_count=s.blade_count,
            rpm=rpm,
        ),
        "curve": export_bladegen_curve(
            d2=s.impeller_d2,
            d1=s.impeller_d1,
            b2=s.impeller_b2,
            beta1=s.beta1,
            beta2=s.beta2,
            blade_count=s.blade_count,
        ),
        "format": "bladegen",
    }


# ---------------------------------------------------------------------------
# BladeGen .bgd export (full format with hub/shroud/blade sections)
# ---------------------------------------------------------------------------

class BladgenBgdRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Q [m³/s]")
    head: float = Field(..., gt=0, description="H [m]")
    rpm: float = Field(..., gt=0, description="RPM")
    n_blade_points: int = Field(40, ge=10, le=100)
    n_span_points: int = Field(8, ge=3, le=20)


@router.post("/geometry/export/bladegen")
def export_bladegen_bgd(req: BladgenBgdRequest) -> dict:
    """Export impeller geometry in ANSYS BladeGen .bgd format.

    Runs 1D sizing and generates full 3D geometry, then converts to
    the .bgd text format with hub/shroud profiles, blade sections in
    cylindrical coordinates, and thickness distribution.

    Args:
        req: Request with operating point and discretisation settings.

    Returns:
        Dict with 'bgd' (file content string) and 'format' = "bgd".
    """
    from hpe.geometry.runner.bladegen_export import export_bladegen

    # Re-use the impeller geometry endpoint to build 3D surfaces
    geom_req = GeometryRequest(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        n_blade_points=req.n_blade_points,
        n_span_points=req.n_span_points,
    )
    geom = get_impeller_geometry(geom_req)

    # We need the SizingResult for metadata; run sizing again (cached)
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    bgd_content = export_bladegen(
        sizing_result=sizing,
        blade_surfaces=geom.blade_surfaces,
        hub_profile=geom.hub_profile,
        shroud_profile=geom.shroud_profile,
        blade_count=geom.blade_count,
    )

    return {"bgd": bgd_content, "format": "bgd"}


# ---------------------------------------------------------------------------
# Export endpoint (unchanged)
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    format: str = Field("step", description="step | stl | iges")


@router.post("/geometry/export")
def export_geometry(req: ExportRequest):
    """Export impeller geometry as STEP, STL, or IGES file."""
    try:
        from hpe.core.models import OperatingPoint
        from hpe.geometry.models import RunnerGeometryParams
        from hpe.geometry.runner.impeller import generate_runner
        from hpe.geometry.runner.export import export_runner
        from hpe.core.enums import GeometryFormat
        from hpe.sizing.meanline import run_sizing
    except ImportError:
        raise HTTPException(status_code=501, detail="CadQuery not installed.")

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)
    params = RunnerGeometryParams.from_sizing_result(sizing)
    runner = generate_runner(params)

    fmt_map = {"step": GeometryFormat.STEP, "stl": GeometryFormat.STL, "iges": GeometryFormat.IGES}
    fmt = fmt_map.get(req.format.lower())
    if fmt is None:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {req.format}")

    ext_map = {"step": ".step", "stl": ".stl", "iges": ".iges"}
    tmp = tempfile.NamedTemporaryFile(suffix=ext_map[req.format.lower()], delete=False)
    tmp.close()
    export_runner(runner, tmp.name, fmt=fmt)

    media_types = {"step": "application/step", "stl": "application/sla", "iges": "application/iges"}
    return FileResponse(
        path=tmp.name,
        filename=f"impeller_Q{req.flow_rate:.4f}_H{req.head:.1f}{ext_map[req.format.lower()]}",
        media_type=media_types.get(req.format.lower(), "application/octet-stream"),
    )


class GltfExportRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    format: str = Field("gltf", description="gltf")


@router.post("/geometry/export/gltf")
def export_gltf(req: GltfExportRequest) -> dict:
    """Export impeller geometry as glTF 2.0 JSON (embedded buffers).

    Returns a minimal glTF structure with one mesh per blade surface (PS/SS),
    encoded as base64 binary buffers. Compatible with Three.js GLTFLoader,
    Blender, and model viewers.
    """
    import base64
    import json
    import struct

    from hpe.core.models import OperatingPoint

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)

    geo_req = GeometryRequest(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        n_blade_points=40,
        n_span_points=12,
    )
    impeller = get_impeller_geometry(geo_req)

    all_positions: list[float] = []
    mesh_primitives: list[dict] = []
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    byte_offset = 0

    for surf in impeller.blade_surfaces:
        for grid in [surf.ps, surf.ss]:
            verts: list[float] = []
            for row in grid:
                for pt in row:
                    verts.extend([pt.x / 1000.0, pt.y / 1000.0, pt.z / 1000.0])  # mm → m
            n_verts = len(verts) // 3
            byte_len = n_verts * 3 * 4  # float32
            all_positions.extend(verts)

            buffer_views.append({
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": byte_len,
                "target": 34962,  # ARRAY_BUFFER
            })
            accessors.append({
                "bufferView": len(buffer_views) - 1,
                "componentType": 5126,  # FLOAT
                "count": n_verts,
                "type": "VEC3",
                "min": [min(verts[j::3]) for j in range(3)],
                "max": [max(verts[j::3]) for j in range(3)],
            })
            mesh_primitives.append({
                "attributes": {"POSITION": len(accessors) - 1},
                "mode": 4,  # TRIANGLES
            })
            byte_offset += byte_len

    raw = struct.pack(f"{len(all_positions)}f", *all_positions)
    b64 = base64.b64encode(raw).decode()

    gltf = {
        "asset": {"version": "2.0", "generator": "HPE v1.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "impeller", "primitives": mesh_primitives}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": byte_offset, "uri": f"data:application/octet-stream;base64,{b64}"}],
    }

    return {"gltf": json.dumps(gltf, separators=(",", ":")), "filename": "impeller.gltf"}


# ---------------------------------------------------------------------------
# IGES export endpoint
# ---------------------------------------------------------------------------

class IGESExportRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Flow rate [m3/s]")
    head: float = Field(..., gt=0, description="Head [m]")
    rpm: float = Field(..., gt=0, description="RPM")


@router.post("/geometry/export/iges")
def export_iges(req: IGESExportRequest):
    """Export impeller geometry as IGES 5.3 file.

    Generates blade surfaces and hub/shroud profiles, then writes them
    as Type 128 (B-Spline Surface) and Type 126 (B-Spline Curve) IGES
    entities. Returns the .igs file as a binary download.
    """
    import numpy as np

    from hpe.geometry.runner.iges_export import write_iges

    # Generate geometry using the existing impeller endpoint logic
    geo_req = GeometryRequest(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        resolution_preset="high",  # use high resolution for CAD export
    )
    impeller = get_impeller_geometry(geo_req)

    # Convert blade surfaces to numpy arrays (shape: n_span x n_chord x 3)
    blade_surfaces: list[dict[str, np.ndarray]] = []
    for surf in impeller.blade_surfaces:
        ps_grid = np.array([
            [[pt.x, pt.y, pt.z] for pt in row]
            for row in surf.ps
        ])
        ss_grid = np.array([
            [[pt.x, pt.y, pt.z] for pt in row]
            for row in surf.ss
        ])
        blade_surfaces.append({"ps": ps_grid, "ss": ss_grid})

    # Convert profiles to numpy arrays
    hub_arr = np.array([[p.x, p.y, p.z] for p in impeller.hub_profile])
    shroud_arr = np.array([[p.x, p.y, p.z] for p in impeller.shroud_profile])

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".igs", delete=False)
    tmp.close()

    write_iges(
        blade_surfaces=blade_surfaces,
        hub_profile=hub_arr,
        shroud_profile=shroud_arr,
        filepath=tmp.name,
        author="HPE",
        description=f"Impeller Q={req.flow_rate:.4f} H={req.head:.1f}",
    )

    return FileResponse(
        path=tmp.name,
        filename=f"impeller_Q{req.flow_rate:.4f}_H{req.head:.1f}.igs",
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Blade loading field (rVθ contours)
# ---------------------------------------------------------------------------

class BladeLoadingRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Volume flow rate [m³/s]")
    head: float = Field(..., gt=0, description="Total head [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [RPM]")
    n_blade_points: int = Field(60, ge=20, le=200)
    n_span_points: int = Field(16, ge=4, le=64)


class BladeLoadingField(BaseModel):
    """Per-blade rVθ field normalised to [0,1] for colormap rendering."""
    ps_rvtheta: List[List[float]]  # [n_span][n_chord] normalised rVθ
    ss_rvtheta: List[List[float]]


class BladeLoadingResponse(BaseModel):
    blade_loading: List[BladeLoadingField]
    rvtheta_min: float  # physical min rVθ [m²/s]
    rvtheta_max: float  # physical max rVθ [m²/s]


@router.post("/geometry/blade_loading_field", response_model=BladeLoadingResponse)
def get_blade_loading_field(req: BladeLoadingRequest) -> BladeLoadingResponse:
    """Compute per-vertex rVθ (angular momentum) on blade surfaces.

    For each blade surface point at (span, chord), the angular momentum is:
        rVθ = r * cu
    where cu is the tangential component of absolute velocity interpolated
    from the inlet/outlet velocity triangles along the meridional coordinate.

    The field is normalised to [0,1] for direct use as a diverging colormap.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    mp = sizing.meridional_profile
    d1 = sizing.impeller_d1
    d2 = sizing.impeller_d2
    b2 = sizing.impeller_b2
    b1 = float(mp.get("b1", b2 * 1.2))
    r1 = d1 / 2.0
    r1_hub = float(mp.get("d1_hub", d1 * 0.35)) / 2.0
    r2 = d2 / 2.0

    n_chord = req.n_blade_points
    n_span = req.n_span_points

    hub_rz, shroud_rz = _meridional_curves(r1, r1_hub, r2, b1, b2, n_chord)

    # Velocity triangle data
    vt = sizing.velocity_triangles
    cu1 = float(vt.get("inlet", {}).get("cu", 0.0))
    cu2 = float(vt.get("outlet", {}).get("cu", 0.0))

    # Build rVθ for each span/chord station
    all_rvtheta: list[list[list[float]]] = []  # [blade][span][chord]

    for _blade in range(sizing.blade_count):
        blade_ps: list[list[float]] = []
        blade_ss: list[list[float]] = []
        for k in range(n_span):
            s = k / max(1, n_span - 1)
            ps_row: list[float] = []
            ss_row: list[float] = []
            for i in range(n_chord):
                t = i / max(1, n_chord - 1)
                # Interpolate r along meridional path for this span
                r_h = hub_rz[i][0]
                r_s = shroud_rz[i][0]
                r = r_h + s * (r_s - r_h)
                # Interpolate cu linearly from inlet to outlet along chord
                cu = cu1 + t * (cu2 - cu1)
                rv_theta = r * cu
                ps_row.append(rv_theta)
                # SS has slightly different loading (lower near LE, higher near TE)
                cu_ss = cu1 + t * (cu2 - cu1) * (1.0 + 0.15 * math.sin(math.pi * t))
                ss_row.append(r * cu_ss)
            blade_ps.append(ps_row)
            blade_ss.append(ss_row)
        all_rvtheta.append([blade_ps, blade_ss])

    # Find global min/max for normalisation
    flat_vals: list[float] = []
    for blade_data in all_rvtheta:
        for surface in blade_data:
            for row in surface:
                flat_vals.extend(row)

    rv_min = min(flat_vals) if flat_vals else 0.0
    rv_max = max(flat_vals) if flat_vals else 1.0
    rv_range = rv_max - rv_min if rv_max > rv_min else 1.0

    # Normalise to [0,1]
    loading_fields: list[BladeLoadingField] = []
    for blade_data in all_rvtheta:
        ps_norm = [
            [round((v - rv_min) / rv_range, 4) for v in row]
            for row in blade_data[0]
        ]
        ss_norm = [
            [round((v - rv_min) / rv_range, 4) for v in row]
            for row in blade_data[1]
        ]
        loading_fields.append(BladeLoadingField(ps_rvtheta=ps_norm, ss_rvtheta=ss_norm))

    return BladeLoadingResponse(
        blade_loading=loading_fields,
        rvtheta_min=round(rv_min, 4),
        rvtheta_max=round(rv_max, 4),
    )
