"""API routes for blade intersection / collision detection.

Provides an endpoint to check impeller blades for geometric
intersections and insufficient clearances.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/geometry", tags=["geometry"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IntersectionCheckRequest(BaseModel):
    """Request body for POST /geometry/check_intersections."""

    flow_rate: float = Field(..., gt=0, description="Design flow rate [m3/s]")
    head: float = Field(..., gt=0, description="Design head [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")
    tolerance_mm: float = Field(0.5, gt=0, description="Clearance tolerance [mm]")
    n_span: int = Field(15, ge=3, le=50, description="Spanwise sample points")
    n_blade_points: int = Field(50, ge=10, le=200, description="Chordwise points")
    override_d2: Optional[float] = Field(None, description="Override D2 [m]")
    override_b2: Optional[float] = Field(None, description="Override b2 [m]")
    override_d1: Optional[float] = Field(None, description="Override D1 [m]")


class ProblemRegionResponse(BaseModel):
    """A region with insufficient clearance."""

    span_fraction: float
    chord_fraction: float
    clearance_mm: float
    blade_pair: list[int]


class CollisionReportResponse(BaseModel):
    """Response schema for collision detection."""

    has_intersection: bool
    intersection_points: list[list[float]]
    min_clearance_mm: float
    problem_regions: list[ProblemRegionResponse]
    blade_hub_clearance_mm: float
    blade_shroud_clearance_mm: float
    n_blades_checked: int
    summary: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/check_intersections", response_model=CollisionReportResponse)
def check_intersections(req: IntersectionCheckRequest) -> CollisionReportResponse:
    """Check impeller blades for intersections and tight clearances.

    Generates blade geometry from sizing parameters, then performs
    pairwise distance checks between adjacent blade surfaces.
    """
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing
    from hpe.geometry.models import RunnerGeometryParams
    from hpe.geometry.runner.blade import generate_blade_profile
    from hpe.geometry.runner.blade_collision import detect_intersections

    # 1. Size the pump
    op = OperatingPoint(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        machine_type=MachineType("centrifugal_pump"),
        override_d2=req.override_d2,
        override_b2=req.override_b2,
        override_d1=req.override_d1,
    )

    try:
        sizing = run_sizing(op)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Sizing failed: {exc}")

    # 2. Build geometry params
    mp = sizing.meridional_profile
    if isinstance(mp, dict):
        d1_hub = mp.get("d1_hub", sizing.impeller_d1 * 0.35)
        b1 = mp.get("b1", sizing.impeller_b2 * 1.2)
    else:
        d1_hub = mp.d1_hub if hasattr(mp, "d1_hub") else sizing.impeller_d1 * 0.35
        b1 = mp.b1 if hasattr(mp, "b1") else sizing.impeller_b2 * 1.2

    params = RunnerGeometryParams(
        d2=sizing.impeller_d2,
        d1=sizing.impeller_d1,
        d1_hub=d1_hub,
        b2=sizing.impeller_b2,
        b1=b1,
        beta1=sizing.beta1,
        beta2=sizing.beta2,
        blade_count=sizing.blade_count,
    )

    # 3. Generate blade profile
    profile = generate_blade_profile(params, n_points=req.n_blade_points)

    # 4. Build blade surfaces for collision detection
    # Each blade = [pressure_side_points, suction_side_points]
    blade_surfaces = [
        [profile.pressure_side, profile.suction_side]
    ]

    # Axial extent
    z_hub = 0.0
    z_shroud = sizing.impeller_b2

    # 5. Run collision detection
    report = detect_intersections(
        blade_surfaces=blade_surfaces,
        n_blades=sizing.blade_count,
        tolerance_mm=req.tolerance_mm,
        z_hub=z_hub,
        z_shroud=z_shroud,
        n_span=req.n_span,
        hub_radius=d1_hub / 2.0,
        shroud_radius=sizing.impeller_d2 / 2.0,
    )

    return CollisionReportResponse(
        has_intersection=report.has_intersection,
        intersection_points=[list(pt) for pt in report.intersection_points],
        min_clearance_mm=report.min_clearance_mm,
        problem_regions=[
            ProblemRegionResponse(
                span_fraction=pr.span_fraction,
                chord_fraction=pr.chord_fraction,
                clearance_mm=pr.clearance_mm,
                blade_pair=list(pr.blade_pair),
            )
            for pr in report.problem_regions
        ],
        blade_hub_clearance_mm=report.blade_hub_clearance_mm,
        blade_shroud_clearance_mm=report.blade_shroud_clearance_mm,
        n_blades_checked=report.n_blades_checked,
        summary=report.summary,
    )
