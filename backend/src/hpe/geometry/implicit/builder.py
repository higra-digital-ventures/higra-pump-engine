"""Build a prototype impeller as an implicit SDF from a sizing result.

This is a *feasibility prototype*, not a production impeller: a shrouded disk
with a central eye bore and ``Z`` radial blade slabs. The point is to show the
free/generative path end-to-end (intent → SDF → voxel field → optional STL),
not to compete with the parametric B-rep geometry for manufacturable detail.
"""

from __future__ import annotations

from typing import Any

from hpe.geometry.implicit import sdf as S

Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def build_impeller_sdf(sizing: Any) -> tuple[S.SDF, dict, Bounds]:
    """Return ``(sdf, meta, bounds)`` for a prototype bladed disk.

    Dimensions are read from the sizing result (metres); sensible fallbacks are
    used when a field is missing so the prototype always produces something.
    """
    d2 = float(getattr(sizing, "impeller_d2", 0.30) or 0.30)
    d1 = float(getattr(sizing, "impeller_d1", 0.45 * d2) or 0.45 * d2)
    b2 = float(getattr(sizing, "impeller_b2", 0.06 * d2) or 0.06 * d2)
    z_blades = int(getattr(sizing, "blade_count", 6) or 6)
    t_blade = float(getattr(sizing, "blade_thickness", 0.0) or 0.04 * d2)

    r2 = d2 / 2.0
    r_eye = max(d1 / 2.0, 0.05 * r2)
    disk_t = max(0.6 * b2, 0.02 * r2)
    blade_h = max(b2, 0.05 * r2)

    # Shrouded disk with a central eye bore.
    disk = S.capped_cylinder_z(r2, 0.0, disk_t)
    bore = S.capped_cylinder_z(r_eye, -disk_t, 2 * disk_t)
    disk_solid = S.subtract(disk, bore)

    # Radial blade slabs standing on the disk, evenly spaced around the axis.
    length = max(r2 - r_eye, 0.1 * r2)
    r_mid = 0.5 * (r_eye + r2)
    half = (length / 2.0, t_blade / 2.0, blade_h / 2.0)
    slab = S.box(half, center=(r_mid, 0.0, disk_t + blade_h / 2.0))
    blades = [S.rotate_z(slab, 2.0 * 3.141592653589793 * i / z_blades) for i in range(z_blades)]

    solid = S.union(disk_solid, *blades)

    total_h = disk_t + blade_h
    pad = 0.15 * r2
    bounds: Bounds = (
        (-r2 - pad, r2 + pad),
        (-r2 - pad, r2 + pad),
        (-0.1 * total_h, total_h + 0.1 * total_h),
    )
    meta = {
        "R2_mm": round(r2 * 1000, 1),
        "R_eye_mm": round(r_eye * 1000, 1),
        "disk_thickness_mm": round(disk_t * 1000, 1),
        "blade_height_mm": round(blade_h * 1000, 1),
        "blade_thickness_mm": round(t_blade * 1000, 2),
        "blade_count": z_blades,
        "total_height_mm": round(total_h * 1000, 1),
    }
    return solid, meta, bounds
