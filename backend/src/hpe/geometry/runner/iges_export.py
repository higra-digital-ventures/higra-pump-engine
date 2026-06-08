"""IGES 5.3 file writer for impeller geometry export.

Writes blade surfaces as Rational B-Spline Surfaces (Type 128) and
hub/shroud profiles as Rational B-Spline Curves (Type 126) in a
standards-compliant IGES file.

The B-spline approximation converts the discrete point grid from the
geometry engine into degree-3 B-spline representations with uniform
knot vectors.

References:
    - IGES 5.3 Specification (ANSI Y14.26M)
    - NURBS Book, Piegl & Tiller, 1997
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class IGESEntity:
    """Represents a single IGES entity (directory entry + parameter data)."""

    entity_type: int
    parameter_lines: list[str] = field(default_factory=list)
    # Directory entry fields
    color: int = 0
    form: int = 0
    label: str = ""


def _uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Generate a clamped uniform knot vector.

    Args:
        n_ctrl: Number of control points.
        degree: B-spline degree.

    Returns:
        Knot vector of length n_ctrl + degree + 1.
    """
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)

    # Clamped: first (degree+1) knots = 0, last (degree+1) = 1
    n_internal = n_knots - 2 * (degree + 1)
    for i in range(n_knots):
        if i <= degree:
            knots[i] = 0.0
        elif i >= n_knots - degree - 1:
            knots[i] = 1.0
        else:
            knots[i] = (i - degree) / (n_internal + 1)

    return knots


def _approximate_bspline_curve(
    points: np.ndarray,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate a set of 3D points with a B-spline curve.

    Uses the point grid directly as control points (interpolation
    for well-spaced data). For blade geometry with smooth distributions,
    this provides adequate approximation.

    Args:
        points: Array of shape (N, 3) with XYZ coordinates.
        degree: B-spline degree (default 3).

    Returns:
        Tuple of (knots, weights, control_points).
    """
    n = len(points)
    if n <= degree:
        degree = max(1, n - 1)

    knots = _uniform_knots(n, degree)
    weights = np.ones(n)
    ctrl = points.copy()

    return knots, weights, ctrl


def _approximate_bspline_surface(
    grid: np.ndarray,
    degree_u: int = 3,
    degree_v: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Approximate a 2D grid of 3D points with a B-spline surface.

    Uses the grid points directly as control points with uniform knots.

    Args:
        grid: Array of shape (nu, nv, 3) with XYZ coordinates.
        degree_u: B-spline degree in u direction.
        degree_v: B-spline degree in v direction.

    Returns:
        Tuple of (knots_u, knots_v, weights, control_points, n_u, n_v).
    """
    nu, nv = grid.shape[0], grid.shape[1]

    if nu <= degree_u:
        degree_u = max(1, nu - 1)
    if nv <= degree_v:
        degree_v = max(1, nv - 1)

    knots_u = _uniform_knots(nu, degree_u)
    knots_v = _uniform_knots(nv, degree_v)
    weights = np.ones((nu, nv))
    ctrl = grid.copy()

    return knots_u, knots_v, weights, ctrl, degree_u, degree_v


def _format_iges_line(content: str, section: str, seq: int) -> str:
    """Format a single 80-character IGES line.

    Args:
        content: The content of the line (up to 72 chars).
        section: Section identifier (S, G, D, P, T).
        seq: Sequence number within the section.

    Returns:
        Formatted 80-character line with newline.
    """
    # Pad content to 72 chars
    line = content.ljust(72)[:72]
    return f"{line}{section}{seq:7d}\n"


def write_iges(
    blade_surfaces: list[dict[str, np.ndarray]],
    hub_profile: np.ndarray,
    shroud_profile: np.ndarray,
    filepath: str,
    author: str = "HPE",
    description: str = "Impeller geometry",
) -> None:
    """Write impeller geometry to an IGES 5.3 file.

    Args:
        blade_surfaces: List of dicts with 'ps' and 'ss' keys,
            each an ndarray of shape (n_span, n_chord, 3) in mm.
        hub_profile: Hub meridional profile, shape (N, 3) in mm.
        shroud_profile: Shroud meridional profile, shape (N, 3) in mm.
        filepath: Output file path.
        author: Author name for the global section.
        description: File description for the start section.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d.%H%M%S")

    entities: list[IGESEntity] = []

    # --- Add blade surfaces as Type 128 (Rational B-Spline Surface) ---
    for b_idx, blade in enumerate(blade_surfaces):
        for side_name, grid in [("PS", blade["ps"]), ("SS", blade["ss"])]:
            if grid.shape[0] < 2 or grid.shape[1] < 2:
                continue

            knots_u, knots_v, weights, ctrl, deg_u, deg_v = (
                _approximate_bspline_surface(grid)
            )
            nu, nv = ctrl.shape[0], ctrl.shape[1]

            entity = IGESEntity(entity_type=128, form=0)
            entity.label = f"B{b_idx}{side_name}"

            # Build parameter data for Type 128
            # Format: 128, K1, K2, M1, M2, PROP1-5, knots_u, knots_v,
            #          weights, control_points, u_start, u_end, v_start, v_end
            k1 = nu - 1  # upper index of first sum
            k2 = nv - 1  # upper index of second sum
            m1 = deg_u
            m2 = deg_v
            prop1 = 0  # not closed in u
            prop2 = 0  # not closed in v
            prop3 = 1  # rational
            prop4 = 0  # not periodic in u
            prop5 = 0  # not periodic in v

            params: list[str] = []
            params.append(f"128,{k1},{k2},{m1},{m2},{prop1},{prop2},{prop3},{prop4},{prop5},")

            # Knot vector u
            kn_u_str = ",".join(f"{v:.6f}" for v in knots_u)
            params.append(f"{kn_u_str},")

            # Knot vector v
            kn_v_str = ",".join(f"{v:.6f}" for v in knots_v)
            params.append(f"{kn_v_str},")

            # Weights (row-major: u varies fastest)
            w_vals: list[str] = []
            for j in range(nv):
                for i in range(nu):
                    w_vals.append(f"{weights[i, j]:.6f}")
            params.append(",".join(w_vals) + ",")

            # Control points (x, y, z for each, row-major)
            for j in range(nv):
                for i in range(nu):
                    x, y, z = ctrl[i, j]
                    params.append(f"{x:.6f},{y:.6f},{z:.6f},")

            # Parameter space bounds
            params.append("0.000000,1.000000,0.000000,1.000000;")

            entity.parameter_lines = params
            entities.append(entity)

    # --- Add hub profile as Type 126 (Rational B-Spline Curve) ---
    for prof_name, profile in [("Hub", hub_profile), ("Shroud", shroud_profile)]:
        if len(profile) < 2:
            continue

        knots, weights_c, ctrl = _approximate_bspline_curve(profile)
        n = len(ctrl)
        degree = min(3, n - 1)

        entity = IGESEntity(entity_type=126, form=0)
        entity.label = prof_name[:8]

        k = n - 1  # upper index of sum
        m = degree
        prop1 = 0  # not planar
        prop2 = 0  # not closed
        prop3 = 1  # rational
        prop4 = 0  # not periodic

        params = []
        params.append(f"126,{k},{m},{prop1},{prop2},{prop3},{prop4},")

        # Knot vector
        kn_str = ",".join(f"{v:.6f}" for v in knots)
        params.append(f"{kn_str},")

        # Weights
        w_str = ",".join(f"{w:.6f}" for w in weights_c)
        params.append(f"{w_str},")

        # Control points
        for i in range(n):
            x, y, z = ctrl[i]
            params.append(f"{x:.6f},{y:.6f},{z:.6f},")

        # Parameter bounds
        params.append("0.000000,1.000000,0.000000,0.000000,0.000000;")

        entity.parameter_lines = params
        entities.append(entity)

    # --- Assemble the IGES file ---
    lines: list[str] = []

    # Start Section
    s_lines = [
        f"HPE Impeller Geometry - {description}",
        f"Generated by Higra Pump Engine on {now.strftime('%Y-%m-%d %H:%M')}",
    ]
    for i, s in enumerate(s_lines, 1):
        lines.append(_format_iges_line(s, "S", i))
    n_start = len(s_lines)

    # Global Section
    g_parts: list[str] = [
        "1H,,",                           # parameter delimiter
        "1H;,",                           # record delimiter
        f"{len(description)}H{description},",  # product ID from sending system
        f"{len(filepath)}H{filepath},",    # file name
        "7HHPE 1.0,",                     # system ID
        "7HHPE 1.0,",                     # preprocessor version
        "32,",                            # int bits
        "75,",                            # SP magnitude
        "15,",                            # SP significance
        "64,",                            # DP magnitude
        "15,",                            # DP significance
        f"{len(description)}H{description},",  # product ID for receiver
        "1.0,",                           # model space scale
        "2,",                             # units flag: mm
        "2HMM,",                          # units name
        "1,",                             # max line weight gradations
        "0.01,",                          # max line width
        f"{len(timestamp)}H{timestamp},",  # file generation timestamp
        "0.001,",                         # min resolution
        "10000.0,",                       # max coordinate
        f"{len(author)}H{author},",        # author
        "26HHigra Industrial Ltda.,",      # organization
        "11,",                            # IGES version (5.3)
        "0,",                             # drafting standard
        f"{len(timestamp)}H{timestamp};",  # model creation timestamp
    ]

    # Build global lines (can span multiple 72-char lines)
    g_text = "".join(g_parts)
    g_lines_list: list[str] = []
    while g_text:
        chunk = g_text[:72]
        g_text = g_text[72:]
        g_lines_list.append(chunk)

    for i, gl in enumerate(g_lines_list, 1):
        lines.append(_format_iges_line(gl, "G", i))
    n_global = len(g_lines_list)

    # Directory Entry and Parameter Data sections
    d_lines: list[str] = []
    p_lines: list[str] = []
    p_seq = 0
    d_seq = 0

    for entity in entities:
        # Flatten parameter lines into 64-char chunks
        # (column 1-64 for data, 65 for space, 66-72 for DE pointer)
        p_text = "".join(entity.parameter_lines)
        p_chunks: list[str] = []
        while p_text:
            chunk = p_text[:64]
            p_text = p_text[64:]
            p_chunks.append(chunk)

        p_start = p_seq + 1
        p_count = len(p_chunks)

        # Write parameter data lines
        de_pointer = d_seq * 2 + 1  # DE sequence number (odd)
        for chunk in p_chunks:
            p_seq += 1
            line_content = f"{chunk.ljust(64)} {de_pointer:7d}"
            p_lines.append(_format_iges_line(line_content, "P", p_seq))

        # Directory entry: two lines per entity
        d_seq_num = d_seq * 2 + 1

        # Line 1 of DE
        de1 = (
            f"{entity.entity_type:8d}"   # entity type
            f"{p_start:8d}"              # parameter data pointer
            f"{0:8d}"                    # structure
            f"{0:8d}"                    # line font pattern
            f"{0:8d}"                    # level
            f"{0:8d}"                    # view
            f"{0:8d}"                    # transformation matrix
            f"{0:8d}"                    # label display assoc
            f"{'00000000':8s}"           # status number
        )
        d_lines.append(_format_iges_line(de1, "D", d_seq_num))

        # Line 2 of DE
        de2 = (
            f"{entity.entity_type:8d}"   # entity type (repeated)
            f"{0:8d}"                    # line weight
            f"{entity.color:8d}"         # color
            f"{p_count:8d}"              # parameter line count
            f"{entity.form:8d}"          # form number
            f"{'':8s}"                   # reserved
            f"{'':8s}"                   # reserved
            f"{entity.label:8s}"         # entity label
            f"{0:8d}"                    # entity subscript
        )
        d_lines.append(_format_iges_line(de2, "D", d_seq_num + 1))

        d_seq += 1

    # Add D and P lines
    lines.extend(d_lines)
    lines.extend(p_lines)

    n_dir = len(d_lines)
    n_param = len(p_lines)

    # Terminate section
    term = f"{n_start:8d}{n_global:8d}{n_dir:8d}{n_param:8d}"
    lines.append(_format_iges_line(term.ljust(72), "T", 1))

    with open(filepath, "w", newline="\r\n") as f:
        f.writelines(lines)
