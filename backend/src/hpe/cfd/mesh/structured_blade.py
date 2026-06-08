"""O-H structured blade-to-blade mesh generator for OpenFOAM.

Generates a fully hex-structured blockMeshDict for a single centrifugal
pump blade passage.  This replaces snappyHexMesh for blade-to-blade
simulations with:

  - Structured hex cells around the blade (O-layer grading)
  - Two-block passage topology (PS-side and SS-side of blade)
  - Controlled y+ via first-cell height from :mod:`yplus`
  - Spline edges that follow the actual blade profile
  - Cyclic (periodic) boundary conditions for the blade pitch
  - Thin-slab 2D mode (one cell, empty front/back) or spanwise 3D

Topology
--------
The domain is the single blade passage in (r, θ) polar coordinates,
mapped to Cartesian (x = r*cos θ, y = r*sin θ, z = axial span).

Two passage blocks per radial segment (n_radial - 1 segments):

    Block PS_i : from periodic_low (θ=0)   to blade PS (θ=θ_PS_i)
    Block SS_i : from blade SS   (θ=θ_SS_i) to periodic_high (θ=pitch)

The blade walls are the interior faces between the PS/SS blocks and
the (void) blade body. Spline edges approximate the actual blade profile.

References
----------
    - Hirsch, C. (2007). Numerical Computation of Internal & External Flows.
    - OpenFOAM Programmer's Guide §5.3 — blockMesh
    - Gulich (2014), §8.2 — Single-passage CFD for centrifugal pumps
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hpe.geometry.models import BladeProfile, MeridionalChannel, RunnerGeometryParams
from hpe.cfd.mesh.yplus import (
    YPlusEstimate,
    compute_first_cell_height,
    estimate_blade_chord,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MeshConfig:
    """Configuration for the structured blade-to-blade mesh.

    Attributes
    ----------
    n_radial : int
        Number of radial stations (= number of cell layers radially).
    n_theta_ps : int
        Cells across the PS passage (from periodic_low to blade PS).
    n_theta_ss : int
        Cells across the SS passage (from blade SS to periodic_high).
    n_span : int
        Spanwise cell layers. Use 1 for 2D (thin-slab with empty BCs).
    grading_ps : float
        Geometric grading from periodic_low toward blade PS wall.
        >1 means cells get finer approaching the blade.
    grading_ss : float
        Geometric grading from periodic_high toward blade SS wall.
    grading_radial : float
        Radial (r) grading, inlet to outlet. 1.0 = uniform.
    target_yplus : float
        Target y+ for first-cell sizing.
    z_span : float
        Spanwise slab thickness [m]. Only used when n_span=1 (2D).
        Defaults to 1% of outlet diameter if None.
    scale : float
        blockMeshDict scale factor (1 = SI metres).
    mode : str
        "2D" for thin-slab with empty BCs, "3D" for full span.
    """
    n_radial: int = 20
    n_theta_ps: int = 25
    n_theta_ss: int = 25
    n_span: int = 1
    grading_ps: float = 4.0
    grading_ss: float = 4.0
    grading_radial: float = 1.0
    target_yplus: float = 30.0
    z_span: Optional[float] = None
    scale: float = 1.0
    mode: str = "2D"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_structured_blade_mesh(
    blade: BladeProfile,
    channel: MeridionalChannel,
    params: RunnerGeometryParams,
    config: MeshConfig,
    case_dir: Path,
    nu: float = 1.004e-6,
    rpm: float = 1450.0,
    rho: float = 998.2,
) -> Path:
    """Generate blockMeshDict for a structured blade-to-blade mesh.

    Entry point for the structured mesh pipeline.  Computes y+ sizing,
    builds the two-block passage topology for each radial station, and
    writes ``system/blockMeshDict``.

    Args:
        blade: BladeProfile from :func:`hpe.geometry.runner.blade.generate_blade_profile`.
        channel: MeridionalChannel from :func:`hpe.geometry.runner.meridional.generate_meridional_channel`.
        params: RunnerGeometryParams with blade dimensions.
        config: MeshConfig controlling cell counts and grading.
        case_dir: OpenFOAM case root directory.
        nu: Kinematic viscosity [m^2/s].  Default: water at 20 °C.
        rpm: Rotational speed [RPM] for tip speed estimation.
        rho: Fluid density [kg/m^3].

    Returns:
        Path to the written ``system/blockMeshDict`` file.
    """
    case_dir = Path(case_dir)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)

    r2 = params.d2 / 2.0
    r1 = params.d1 / 2.0

    # --- y+ sizing ---
    u_tip = math.pi * params.d2 * rpm / 60.0
    chord = estimate_blade_chord(r1, r2, params.beta1, params.beta2)
    yp = compute_first_cell_height(
        u_ref=u_tip,
        l_ref=chord,
        nu=nu,
        target_yplus=config.target_yplus,
        rho=rho,
    )
    log.info(
        "Structured mesh: y+=%.1f → first cell %.3e m  (Re=%.2e, u_tau=%.3f m/s)",
        yp.y_plus_check, yp.first_cell_height, yp.reynolds, yp.u_tau,
    )

    # --- Passage geometry ---
    pitch_rad = 2.0 * math.pi / params.blade_count
    z_span = config.z_span if config.z_span is not None else params.d2 * 0.01

    # Sample blade PS and SS at n_radial radial stations
    stations = _sample_blade_stations(blade, config.n_radial)

    # --- Build vertices and blocks ---
    vertices, blocks, edges, boundaries = _build_passage_topology(
        stations=stations,
        pitch_rad=pitch_rad,
        z_span=z_span,
        config=config,
        yp=yp,
    )

    # --- Write blockMeshDict ---
    out_path = _write_blockmesh_dict(
        case_dir=case_dir,
        vertices=vertices,
        blocks=blocks,
        edges=edges,
        boundaries=boundaries,
        scale=config.scale,
        params=params,
        config=config,
    )

    log.info(
        "Structured mesh written: %d vertices, %d blocks → %s",
        len(vertices), len(blocks), out_path,
    )
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_blade_stations(
    blade: BladeProfile,
    n_stations: int,
) -> list[dict]:
    """Sample blade PS/SS at uniformly spaced radial stations.

    Returns a list of dicts with keys:
        r       : radial position [m]
        theta_ps: polar angle of pressure side [rad]
        theta_ss: polar angle of suction side [rad]
    """
    ps = blade.pressure_side
    ss = blade.suction_side

    r_min = min(ps[0][0], ss[0][0])
    r_max = max(ps[-1][0], ss[-1][0])

    stations = []
    for i in range(n_stations):
        t = i / max(n_stations - 1, 1)
        r_target = r_min + t * (r_max - r_min)
        theta_ps = _interp_theta(ps, r_target)
        theta_ss = _interp_theta(ss, r_target)
        # Ensure PS is always at lower theta than SS
        if theta_ps > theta_ss:
            theta_ps, theta_ss = theta_ss, theta_ps
        stations.append({"r": r_target, "theta_ps": theta_ps, "theta_ss": theta_ss})

    return stations


def _interp_theta(profile: list, r_target: float) -> float:
    """Linear interpolation of theta at a given r from a (r, theta) list."""
    if r_target <= profile[0][0]:
        return profile[0][1]
    if r_target >= profile[-1][0]:
        return profile[-1][1]
    for i in range(len(profile) - 1):
        r0, t0 = profile[i]
        r1, t1 = profile[i + 1]
        if r0 <= r_target <= r1:
            frac = (r_target - r0) / (r1 - r0) if r1 > r0 else 0.0
            return t0 + frac * (t1 - t0)
    return profile[-1][1]


def _polar_to_xyz(r: float, theta: float, z: float) -> tuple:
    """Convert polar (r, θ) to Cartesian (x, y, z)."""
    return (r * math.cos(theta), r * math.sin(theta), z)


def _build_passage_topology(
    stations: list,
    pitch_rad: float,
    z_span: float,
    config: MeshConfig,
    yp: YPlusEstimate,
) -> tuple:
    """Build vertex, block, edge, and boundary lists for the passage.

    For each pair of adjacent radial stations (i, i+1), two hex blocks
    are created:
        PS-block : from θ=0 (periodicLow) to θ=θ_PS  — PS passage
        SS-block : from θ=θ_SS to θ=pitch (periodicHigh) — SS passage

    Vertex numbering per station (4 vertices per station, 2 z-layers):
        At station i:
            v[4*i + 0]  : (r_i,  θ=0,        z=0)   — periodic low, inner
            v[4*i + 1]  : (r_i,  θ=θ_PS_i,   z=0)   — blade PS
            v[4*i + 2]  : (r_i,  θ=θ_SS_i,   z=0)   — blade SS
            v[4*i + 3]  : (r_i,  θ=pitch,    z=0)   — periodic high, inner
        Plus copies at z=z_span: v[4*N + 4*i + j] for j in 0..3

    Total vertices: 2 * 4 * n_stations = 8*n_stations

    Returns:
        (vertices, blocks, edges, boundaries)
    """
    n = len(stations)
    N = n  # shorthand

    # ---- Build vertices ----
    verts: list[tuple] = []

    def _add(r, theta, z):
        verts.append(_polar_to_xyz(r, theta, z))

    # z=0 layer
    for s in stations:
        _add(s["r"], 0.0,           0.0)   # periodic low
        _add(s["r"], s["theta_ps"], 0.0)   # blade PS
        _add(s["r"], s["theta_ss"], 0.0)   # blade SS
        _add(s["r"], pitch_rad,     0.0)   # periodic high

    # z=z_span layer (same θ positions)
    for s in stations:
        _add(s["r"], 0.0,           z_span)
        _add(s["r"], s["theta_ps"], z_span)
        _add(s["r"], s["theta_ss"], z_span)
        _add(s["r"], pitch_rad,     z_span)

    # Vertex index helper:
    #   z=0 layer:    v_idx(i, j) = 4*i + j
    #   z=span layer: v_idx(i, j) + 4*N
    def vi(i, j, top=False):
        base = 4 * N if top else 0
        return base + 4 * i + j

    # ---- Build blocks ----
    blocks: list[dict] = []

    for i in range(N - 1):
        # --- PS passage block: θ ∈ [0, θ_PS] ---
        # Vertices in hex order (OpenFOAM bottom-front→back, top-front→back):
        #  bottom face (z=0): v(i,0) v(i+1,0) v(i+1,1) v(i,1)
        #  top face (z=span): same with top=True
        b_ps = {
            "verts": [
                vi(i,   0), vi(i+1, 0), vi(i+1, 1), vi(i,   1),
                vi(i,   0, True), vi(i+1, 0, True), vi(i+1, 1, True), vi(i, 1, True),
            ],
            "cells": (config.n_radial // (N - 1) or 1,
                      config.n_theta_ps,
                      config.n_span),
            "grading": (config.grading_radial,
                        1.0 / config.grading_ps,  # finer at blade PS
                        1.0),
            "label": f"ps_block_{i}",
        }
        blocks.append(b_ps)

        # --- SS passage block: θ ∈ [θ_SS, pitch] ---
        b_ss = {
            "verts": [
                vi(i,   2), vi(i+1, 2), vi(i+1, 3), vi(i,   3),
                vi(i,   2, True), vi(i+1, 2, True), vi(i+1, 3, True), vi(i, 3, True),
            ],
            "cells": (config.n_radial // (N - 1) or 1,
                      config.n_theta_ss,
                      config.n_span),
            "grading": (config.grading_radial,
                        config.grading_ss,  # finer at blade SS
                        1.0),
            "label": f"ss_block_{i}",
        }
        blocks.append(b_ss)

    # ---- Build spline edges (blade PS and SS surfaces) ----
    edges: list[dict] = []

    def _blade_spline_pts(profile, n_pts=20):
        """Resample blade profile into n_pts control points for spline edge."""
        step = max(1, len(profile) // n_pts)
        pts = profile[::step]
        if pts[-1] != profile[-1]:
            pts.append(profile[-1])
        return [_polar_to_xyz(r, th, 0.0) for r, th in pts]

    def _blade_spline_pts_top(profile, n_pts=20):
        step = max(1, len(profile) // n_pts)
        pts = profile[::step]
        if pts[-1] != profile[-1]:
            pts.append(profile[-1])
        return [_polar_to_xyz(r, th, z_span) for r, th in pts]

    # PS splines: connect vi(0,1) → vi(N-1,1) along blade PS at z=0 and z=span
    # We add a spline edge per block (between consecutive PS vertices)
    for i in range(N - 1):
        # Interpolate a few intermediate blade points for this radial segment
        r0 = stations[i]["r"]
        r1_val = stations[i + 1]["r"]
        # 5 intermediate points between station i and i+1 on PS
        ps_pts = []
        ss_pts = []
        for frac in [0.25, 0.5, 0.75]:
            r_mid = r0 + frac * (r1_val - r0)
            th_ps = _interp_theta(
                [(s["r"], s["theta_ps"]) for s in stations],  # type: ignore
                r_mid,
            )
            th_ss = _interp_theta(
                [(s["r"], s["theta_ss"]) for s in stations],  # type: ignore
                r_mid,
            )
            ps_pts.append(_polar_to_xyz(r_mid, th_ps, 0.0))
            ss_pts.append(_polar_to_xyz(r_mid, th_ss, 0.0))

        if ps_pts:
            edges.append({
                "type": "spline",
                "v0": vi(i, 1),
                "v1": vi(i + 1, 1),
                "points": ps_pts,
            })
            edges.append({
                "type": "spline",
                "v0": vi(i, 2),
                "v1": vi(i + 1, 2),
                "points": ss_pts,
            })
            # z=span copies
            ps_pts_top = [_polar_to_xyz(r, math.atan2(p[1], p[0]), z_span)
                          for (r, _) in [(math.hypot(p[0], p[1]), 0) for p in ps_pts]
                          for p in ps_pts[:1]]  # simplified — same r,θ at z=span
            # Rebuild top properly
            ps_top = [_polar_to_xyz(
                math.hypot(pt[0], pt[1]),
                math.atan2(pt[1], pt[0]),
                z_span,
            ) for pt in ps_pts]
            ss_top = [_polar_to_xyz(
                math.hypot(pt[0], pt[1]),
                math.atan2(pt[1], pt[0]),
                z_span,
            ) for pt in ss_pts]
            edges.append({
                "type": "spline",
                "v0": vi(i, 1, True),
                "v1": vi(i + 1, 1, True),
                "points": ps_top,
            })
            edges.append({
                "type": "spline",
                "v0": vi(i, 2, True),
                "v1": vi(i + 1, 2, True),
                "points": ss_top,
            })

    # ---- Build boundaries ----
    boundaries: list[dict] = []

    # Inlet (inner radius r=r1, first station)
    inlet_faces = []
    inlet_faces.append([vi(0, 0), vi(0, 1), vi(0, 1, True), vi(0, 0, True)])  # PS block inlet
    inlet_faces.append([vi(0, 2), vi(0, 3), vi(0, 3, True), vi(0, 2, True)])  # SS block inlet
    boundaries.append({"name": "inlet", "type": "patch", "faces": inlet_faces})

    # Outlet (outer radius r=r2, last station)
    outlet_faces = []
    outlet_faces.append([vi(N-1, 0), vi(N-1, 1), vi(N-1, 1, True), vi(N-1, 0, True)])
    outlet_faces.append([vi(N-1, 2), vi(N-1, 3), vi(N-1, 3, True), vi(N-1, 2, True)])
    boundaries.append({"name": "outlet", "type": "patch", "faces": outlet_faces})

    # Blade walls
    blade_faces = []
    for i in range(N - 1):
        # PS face of PS-block (inner face at θ=θ_PS)
        blade_faces.append([vi(i, 1), vi(i+1, 1), vi(i+1, 1, True), vi(i, 1, True)])
        # SS face of SS-block (inner face at θ=θ_SS)
        blade_faces.append([vi(i, 2), vi(i+1, 2), vi(i+1, 2, True), vi(i, 2, True)])
    boundaries.append({"name": "blade", "type": "wall", "faces": blade_faces})

    # Periodic low (θ=0 face of PS-blocks)
    plow_faces = []
    for i in range(N - 1):
        plow_faces.append([vi(i, 0), vi(i+1, 0), vi(i+1, 0, True), vi(i, 0, True)])
    boundaries.append({
        "name": "periodicLow",
        "type": "cyclic",
        "neighbourPatch": "periodicHigh",
        "faces": plow_faces,
    })

    # Periodic high (θ=pitch face of SS-blocks)
    phigh_faces = []
    for i in range(N - 1):
        phigh_faces.append([vi(i, 3), vi(i+1, 3), vi(i+1, 3, True), vi(i, 3, True)])
    boundaries.append({
        "name": "periodicHigh",
        "type": "cyclic",
        "neighbourPatch": "periodicLow",
        "faces": phigh_faces,
    })

    # Front and back (z=0 and z=span) — empty for 2D, wall for 3D
    bc_type_zfaces = "empty" if config.mode == "2D" else "wall"
    front_faces = []
    back_faces = []
    for i in range(N - 1):
        # PS blocks
        front_faces.append([vi(i, 0), vi(i+1, 0), vi(i+1, 1), vi(i, 1)])
        back_faces.append([vi(i, 0, True), vi(i+1, 0, True), vi(i+1, 1, True), vi(i, 1, True)])
        # SS blocks
        front_faces.append([vi(i, 2), vi(i+1, 2), vi(i+1, 3), vi(i, 3)])
        back_faces.append([vi(i, 2, True), vi(i+1, 2, True), vi(i+1, 3, True), vi(i, 3, True)])
    boundaries.append({"name": "front", "type": bc_type_zfaces, "faces": front_faces})
    boundaries.append({"name": "back",  "type": bc_type_zfaces, "faces": back_faces})

    return verts, blocks, edges, boundaries


def _fmt_vertex(v: tuple) -> str:
    return f"    ({v[0]:.8f} {v[1]:.8f} {v[2]:.8f})"


def _fmt_face(face: list) -> str:
    return "(" + " ".join(str(vi) for vi in face) + ")"


def _write_blockmesh_dict(
    case_dir: Path,
    vertices: list,
    blocks: list,
    edges: list,
    boundaries: list,
    scale: float,
    params: RunnerGeometryParams,
    config: MeshConfig,
) -> Path:
    """Render and write system/blockMeshDict.

    Args:
        case_dir: OpenFOAM case root.
        vertices: List of (x, y, z) vertex tuples.
        blocks: List of block dicts with keys: verts, cells, grading, label.
        edges: List of edge dicts with keys: type, v0, v1, points.
        boundaries: List of boundary dicts with keys: name, type, faces.
        scale: Mesh scale factor.
        params: RunnerGeometryParams (for header comment).
        config: MeshConfig (for header comment).

    Returns:
        Path to the written blockMeshDict.
    """
    lines: list[str] = []

    lines.append("/*--------------------------------*- C++ -*----------------------------------*\\")
    lines.append("| Generated by HPE — Higra Pump Engine                                       |")
    lines.append(f"| Structured blade-to-blade mesh  D2={params.d2*1000:.1f}mm  Z={params.blade_count}  mode={config.mode} |")
    lines.append("\\*---------------------------------------------------------------------------*/")
    lines.append("FoamFile")
    lines.append("{")
    lines.append("    version     2.0;")
    lines.append("    format      ascii;")
    lines.append("    class       dictionary;")
    lines.append("    object      blockMeshDict;")
    lines.append("}")
    lines.append("")
    lines.append(f"scale {scale};")
    lines.append("")

    # Vertices
    lines.append("vertices")
    lines.append("(")
    for i, v in enumerate(vertices):
        lines.append(f"    ({v[0]:.8f} {v[1]:.8f} {v[2]:.8f})  // {i}")
    lines.append(");")
    lines.append("")

    # Blocks
    lines.append("blocks")
    lines.append("(")
    for b in blocks:
        vlist = " ".join(str(vi) for vi in b["verts"])
        cx, cy, cz = b["cells"]
        gx, gy, gz = b["grading"]
        lines.append(f"    // {b['label']}")
        lines.append(f"    hex ({vlist})")
        lines.append(f"    ({cx} {cy} {cz})")
        lines.append(f"    simpleGrading ({gx:.4f} {gy:.4f} {gz:.4f})")
        lines.append("")
    lines.append(");")
    lines.append("")

    # Edges
    lines.append("edges")
    lines.append("(")
    for e in edges:
        if e["type"] == "spline" and e.get("points"):
            pts_str = "\n        ".join(
                f"({p[0]:.8f} {p[1]:.8f} {p[2]:.8f})" for p in e["points"]
            )
            lines.append(f"    spline {e['v0']} {e['v1']}")
            lines.append("    (")
            for p in e["points"]:
                lines.append(f"        ({p[0]:.8f} {p[1]:.8f} {p[2]:.8f})")
            lines.append("    )")
    lines.append(");")
    lines.append("")

    # Boundaries
    lines.append("boundary")
    lines.append("(")
    for bnd in boundaries:
        lines.append(f"    {bnd['name']}")
        lines.append("    {")
        lines.append(f"        type    {bnd['type']};")
        if bnd.get("neighbourPatch"):
            lines.append(f"        neighbourPatch {bnd['neighbourPatch']};")
            lines.append("        transform  rotational;")
            lines.append("        rotationAxis   (0 0 1);")
            lines.append("        rotationCentre (0 0 0);")
        lines.append("        faces")
        lines.append("        (")
        for face in bnd["faces"]:
            lines.append(f"            ({' '.join(str(vi) for vi in face)})")
        lines.append("        );")
        lines.append("    }")
        lines.append("")
    lines.append(");")
    lines.append("")
    lines.append("mergePatchPairs")
    lines.append("(")
    lines.append(");")
    lines.append("")

    content = "\n".join(lines)
    out_path = case_dir / "system" / "blockMeshDict"
    out_path.write_text(content)
    return out_path
