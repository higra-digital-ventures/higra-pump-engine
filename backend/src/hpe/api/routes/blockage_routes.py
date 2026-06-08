"""Blockage tables API routes.

Endpoints:
    POST /api/v1/physics/blockage         — compute default blockage from blade geometry
    POST /api/v1/physics/blockage/custom   — apply custom blockage table
    GET  /api/v1/physics/blockage/presets  — list available presets
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["physics"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BlockageDefaultRequest(BaseModel):
    """Input for default blockage computation from blade geometry."""

    blade_count: int = Field(..., ge=1, le=50, description="Number of blades z")
    thickness_dist: List[float] = Field(
        ..., min_length=2, description="Blade thickness [m] at meridional stations"
    )
    blade_angles: List[float] = Field(
        ..., min_length=2, description="Blade angle [deg] at meridional stations"
    )
    radii: Optional[List[float]] = Field(
        None, description="Local radius [m] at each station (auto if omitted)"
    )
    n_pts: int = Field(21, ge=3, le=200, description="Output meridional resolution")
    preset: Optional[str] = Field(None, description="Use a named preset instead")


class BlockageCustomRequest(BaseModel):
    """Input for a user-specified blockage table."""

    mode: str = Field(
        "two_point",
        description="'two_point' for linear LE/TE interpolation, 'table' for full 2-D",
    )
    # two_point mode
    b_inlet: Optional[float] = Field(None, ge=0.01, le=1.0, description="Blockage at LE")
    b_outlet: Optional[float] = Field(None, ge=0.01, le=1.0, description="Blockage at TE")
    # table mode
    m_points: Optional[List[float]] = Field(None, description="Meridional coords [0..1]")
    s_points: Optional[List[float]] = Field(None, description="Spanwise coords [0..1]")
    values: Optional[List[List[float]]] = Field(None, description="2-D blockage values")
    n_pts: int = Field(21, ge=3, le=200, description="Output points (two_point mode)")


class BlockageResponse(BaseModel):
    """Blockage table output."""

    m_points: List[float]
    s_points: List[float]
    values: List[List[float]]
    b_inlet: float
    b_outlet: float


class PresetInfo(BaseModel):
    name: str
    b_inlet: float
    b_outlet: float


class PresetsResponse(BaseModel):
    presets: List[PresetInfo]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/physics/blockage", response_model=BlockageResponse)
def compute_blockage(req: BlockageDefaultRequest) -> BlockageResponse:
    """Compute default blockage from blade geometry or use a preset."""
    from hpe.physics.blockage_tables import BlockageTable, PRESETS

    if req.preset:
        if req.preset not in PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset '{req.preset}'. Available: {list(PRESETS.keys())}",
            )
        tbl = PRESETS[req.preset]
    else:
        if len(req.thickness_dist) != len(req.blade_angles):
            raise HTTPException(
                status_code=422,
                detail="thickness_dist and blade_angles must have the same length",
            )
        if req.radii is not None and len(req.radii) != len(req.thickness_dist):
            raise HTTPException(
                status_code=422,
                detail="radii must have the same length as thickness_dist",
            )
        try:
            tbl = BlockageTable.from_default(
                blade_count=req.blade_count,
                thickness_dist=req.thickness_dist,
                blade_angles=req.blade_angles,
                radii=req.radii,
                n_pts=req.n_pts,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    d = tbl.as_dict()
    return BlockageResponse(
        m_points=d["m_points"],
        s_points=d["s_points"],
        values=d["values"],
        b_inlet=float(tbl.interpolate(0.0)),
        b_outlet=float(tbl.interpolate(1.0)),
    )


@router.post("/physics/blockage/custom", response_model=BlockageResponse)
def custom_blockage(req: BlockageCustomRequest) -> BlockageResponse:
    """Apply a custom blockage table (two-point or full 2-D)."""
    from hpe.physics.blockage_tables import BlockageTable

    if req.mode == "two_point":
        if req.b_inlet is None or req.b_outlet is None:
            raise HTTPException(
                status_code=422,
                detail="b_inlet and b_outlet are required for two_point mode",
            )
        tbl = BlockageTable.from_two_control_points(
            req.b_inlet, req.b_outlet, n_pts=req.n_pts,
        )
    elif req.mode == "table":
        if req.m_points is None or req.s_points is None or req.values is None:
            raise HTTPException(
                status_code=422,
                detail="m_points, s_points and values are required for table mode",
            )
        try:
            tbl = BlockageTable.from_table(req.m_points, req.s_points, req.values)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{req.mode}'. Use 'two_point' or 'table'.",
        )

    d = tbl.as_dict()
    return BlockageResponse(
        m_points=d["m_points"],
        s_points=d["s_points"],
        values=d["values"],
        b_inlet=float(tbl.interpolate(0.0)),
        b_outlet=float(tbl.interpolate(1.0)),
    )


@router.get("/physics/blockage/presets", response_model=PresetsResponse)
def list_blockage_presets() -> PresetsResponse:
    """List available blockage presets."""
    from hpe.physics.blockage_tables import PRESETS

    items = []
    for name, tbl in PRESETS.items():
        items.append(PresetInfo(
            name=name,
            b_inlet=float(tbl.interpolate(0.0)),
            b_outlet=float(tbl.interpolate(1.0)),
        ))
    return PresetsResponse(presets=items)
