"""LE/TE refinement API routes.

Exposes leading-edge and trailing-edge modification capabilities
from hpe.geometry.runner.lete_modification through a REST API,
including recommended defaults based on specific speed (Nq).
"""

from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["blade-lete"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LETERequest(BaseModel):
    """Input for LE/TE modification."""

    le_radius_mm: float = Field(..., gt=0, le=10, description="Leading edge radius [mm]")
    te_radius_mm: float = Field(..., gt=0, le=5, description="Trailing edge radius [mm]")
    le_type: Literal["elliptic", "circular", "sharp"] = Field(
        "elliptic", description="Leading edge shape type"
    )
    te_type: Literal["blunt", "tapered", "circular"] = Field(
        "blunt", description="Trailing edge shape type"
    )
    # Sizing reference to generate the blade surface
    flow_rate: float = Field(..., gt=0, description="Flow rate [m3/s]")
    head: float = Field(..., gt=0, description="Head [m]")
    rpm: float = Field(..., gt=0, description="RPM")


class EdgePoint2D(BaseModel):
    """A single point in the 2D edge profile cross-section."""

    x: float
    y: float


class LETEResponse(BaseModel):
    """Result of LE/TE modification."""

    le_thickness_mm: float
    te_thickness_mm: float
    le_radius_mm: float
    te_radius_mm: float
    le_type: str
    te_type: str
    # SVG-friendly 2D cross-section points for before/after preview
    before_le_profile: List[EdgePoint2D]
    after_le_profile: List[EdgePoint2D]
    before_te_profile: List[EdgePoint2D]
    after_te_profile: List[EdgePoint2D]


class LETEDefaults(BaseModel):
    """Recommended LE/TE parameters for a given Nq range."""

    nq: float
    nq_range: str
    le_radius_mm: float
    le_type: str
    te_radius_mm: float
    te_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LE_TYPE_TO_RATIO = {"elliptic": 2.0, "circular": 1.0, "sharp": 4.0}
_TE_TYPE_TO_RATIO = {"blunt": 1.0, "tapered": 2.5, "circular": 1.5}


def _edge_profile_points(
    radius_mm: float,
    edge_type: str,
    n_pts: int = 30,
    is_le: bool = True,
) -> List[EdgePoint2D]:
    """Generate a 2D cross-section profile for an edge shape.

    The profile is a half-ellipse (or circle/sharp) centred at (0, 0),
    suitable for SVG rendering.

    Args:
        radius_mm: Edge radius in mm.
        edge_type: Shape type string.
        n_pts: Number of points to generate.
        is_le: True for leading edge, False for trailing edge.

    Returns:
        List of 2D points forming the edge cross-section.
    """
    import math

    ratio_map = _LE_TYPE_TO_RATIO if is_le else _TE_TYPE_TO_RATIO
    ratio = ratio_map.get(edge_type, 1.5)

    # Semi-axes: b = radius_mm, a = ratio * b (along flow direction)
    b = radius_mm
    a = ratio * b

    points: list[EdgePoint2D] = []
    for i in range(n_pts):
        t = i / (n_pts - 1)  # 0 -> 1
        angle = math.pi * t  # 0 -> pi (half-ellipse)
        x = -a * math.cos(angle) if is_le else a * math.cos(angle)
        y = b * math.sin(angle)
        points.append(EdgePoint2D(x=round(x, 4), y=round(y, 4)))

    return points


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/blade/lete", response_model=LETEResponse)
def apply_lete(req: LETERequest) -> LETEResponse:
    """Apply LE/TE modifications and return modified edge profiles.

    Uses the sizing result to generate a baseline blade profile, then
    applies the requested LE/TE modifications via the geometry module.
    Returns 2D cross-section profiles for before/after SVG preview.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.geometry.runner.lete_modification import (
        LETESpec,
        apply_lete_modification,
    )
    from hpe.geometry.models import BladeProfile

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    # Build a simple representative blade profile for the LE/TE modification
    import math

    d2 = sizing.impeller_d2
    d1 = sizing.impeller_d1
    blade_thickness = max(0.002, d2 * 0.02)
    r1 = d1 / 2.0
    r2 = d2 / 2.0
    beta1 = math.radians(sizing.beta1)
    beta2 = math.radians(sizing.beta2)

    n_pts = 40
    camber: list[tuple[float, float]] = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        r = r1 + t * (r2 - r1)
        beta = beta1 + t * (beta2 - beta1)
        theta = 0.0
        if i > 0:
            dr = r - camber[-1][0]
            r_mid = (r + camber[-1][0]) / 2
            if abs(math.tan(beta)) > 1e-10 and r_mid > 1e-6:
                theta = camber[-1][1] + dr / (r_mid * math.tan(beta))
        camber.append((r, theta))

    # Build PS/SS with uniform thickness
    ps: list[tuple[float, float]] = []
    ss: list[tuple[float, float]] = []
    for r, theta in camber:
        dt = blade_thickness / (2.0 * r) if r > 1e-6 else 0.0
        ps.append((r, theta + dt))
        ss.append((r, theta - dt))

    profile = BladeProfile(
        camber_points=camber,
        pressure_side=ps,
        suction_side=ss,
        thickness=blade_thickness,
    )

    # Map edge types to elliptic ratios
    le_ratio = _LE_TYPE_TO_RATIO.get(req.le_type, 2.0)
    te_ratio = _TE_TYPE_TO_RATIO.get(req.te_type, 1.5)

    spec = LETESpec(
        le_enabled=True,
        le_elliptic_ratio=le_ratio,
        le_extent=0.12,
        te_enabled=True,
        te_elliptic_ratio=te_ratio,
        te_extent=0.12,
        min_edge_thickness=req.te_radius_mm / 1000.0,
    )

    result = apply_lete_modification(profile, spec)

    # Generate before/after 2D edge profiles for SVG preview
    # "Before" = default circular with blade_thickness/4 radius
    default_le_r = blade_thickness * 1000 * 0.25
    default_te_r = blade_thickness * 1000 * 0.167

    before_le = _edge_profile_points(default_le_r, "circular", is_le=True)
    after_le = _edge_profile_points(req.le_radius_mm, req.le_type, is_le=True)
    before_te = _edge_profile_points(default_te_r, "blunt", is_le=False)
    after_te = _edge_profile_points(req.te_radius_mm, req.te_type, is_le=False)

    return LETEResponse(
        le_thickness_mm=result.le_thickness * 1000,
        te_thickness_mm=result.te_thickness * 1000,
        le_radius_mm=req.le_radius_mm,
        te_radius_mm=req.te_radius_mm,
        le_type=req.le_type,
        te_type=req.te_type,
        before_le_profile=before_le,
        after_le_profile=after_le,
        before_te_profile=before_te,
        after_te_profile=after_te,
    )


@router.get("/blade/lete/defaults", response_model=LETEDefaults)
def get_lete_defaults(
    nq: float = Query(..., gt=0, description="Specific speed Nq"),
) -> LETEDefaults:
    """Return recommended LE/TE parameters based on specific speed Nq.

    Ranges:
        Low Nq (< 25):    LE circular r=1.5mm, TE blunt r=0.5mm
        Medium Nq (25-60): LE elliptic r=2mm, TE tapered r=0.8mm
        High Nq (> 60):    LE elliptic r=3mm, TE circular r=1mm
    """
    if nq < 25:
        return LETEDefaults(
            nq=nq,
            nq_range="low (< 25)",
            le_radius_mm=1.5,
            le_type="circular",
            te_radius_mm=0.5,
            te_type="blunt",
        )
    elif nq <= 60:
        return LETEDefaults(
            nq=nq,
            nq_range="medium (25-60)",
            le_radius_mm=2.0,
            le_type="elliptic",
            te_radius_mm=0.8,
            te_type="tapered",
        )
    else:
        return LETEDefaults(
            nq=nq,
            nq_range="high (> 60)",
            le_radius_mm=3.0,
            le_type="elliptic",
            te_radius_mm=1.0,
            te_type="circular",
        )
