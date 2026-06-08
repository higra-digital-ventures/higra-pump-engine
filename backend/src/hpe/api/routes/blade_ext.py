"""Blade geometry API: splitter blades, NACA thickness, stacking.

Endpoints:
    POST /api/v1/blade/splitter    — size splitter blades
    POST /api/v1/blade/thickness   — compute NACA or elliptical thickness
    POST /api/v1/blade/stacking    — compute blade stacking parameters
    GET  /api/v1/blade/wrap_angle  — estimate wrap angle from geometry
"""
from __future__ import annotations
from typing import Literal, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from hpe.geometry.runner.splitter import size_splitter
from hpe.geometry.blade.naca_thickness import (
    ThicknessType,
    get_thickness,
)
from hpe.geometry.blade.stacking import StackingConfig, compute_stacking, wrap_angle_from_geometry

router = APIRouter(prefix="/api/v1/blade", tags=["blade"])


# ── Splitter ──────────────────────────────────────────────────────────────────

class SplitterRequest(BaseModel):
    blade_count: int = Field(..., ge=3, le=12)
    d2: float = Field(..., gt=0)
    d1: float = Field(..., gt=0)
    b2: float = Field(..., gt=0)
    beta2: float = Field(..., gt=0, lt=90)
    start_fraction: float = Field(0.4, ge=0.2, le=0.7)
    pitch_offset: float = Field(0.5, ge=0.0, le=1.0)
    work_ratio: float = Field(0.5, ge=0.3, le=0.7)


class SplitterResponse(BaseModel):
    enabled: bool
    start_fraction: float
    splitter_blade_count: int
    throat_area_reduction: float
    loading_reduction_percent: float
    passage_width_mm: float
    splitter_start_m: float
    warnings: list[str]


@router.post("/splitter", response_model=SplitterResponse)
def size_splitter_endpoint(req: SplitterRequest) -> SplitterResponse:
    """Size splitter blades for a centrifugal impeller."""
    r = size_splitter(
        blade_count=req.blade_count, d2=req.d2, d1=req.d1, b2=req.b2,
        beta2=req.beta2, start_fraction=req.start_fraction,
        pitch_offset=req.pitch_offset, work_ratio=req.work_ratio,
    )
    return SplitterResponse(**r.__dict__)


# ── Thickness ─────────────────────────────────────────────────────────────────

class ThicknessRequest(BaseModel):
    thickness_type: str = Field(
        'naca_4digit',
        description=(
            "Thickness distribution type: naca_4digit, elliptic, biparabolic, "
            "linear_taper, constant, dca, wedge"
        ),
    )
    # Legacy field — mapped to thickness_type for backward compatibility
    profile_type: Optional[Literal['naca', 'ellipse']] = Field(
        None,
        description="DEPRECATED: use thickness_type instead",
    )
    t_max_frac: float = Field(0.08, ge=0.01, le=0.30)
    n_points: int = Field(21, ge=5, le=101)
    # NACA params
    close_te: bool = True
    # Ellipse params
    le_ratio: float = Field(2.0, ge=0.5, le=5.0)
    te_ratio: float = Field(1.0, ge=0.1, le=3.0)
    # Biparabolic / DCA params
    s_max: float = Field(0.3, ge=0.05, le=0.95, description="Location of max thickness (0-1)")
    # Linear taper params
    t_min_frac: float = Field(0.02, ge=0.0, le=0.30, description="TE thickness for linear_taper")


class ThicknessResponse(BaseModel):
    thickness_type: str
    m_normalized: list[float]
    t_normalized: list[float]
    t_max_over_chord: float
    leading_edge_radius: float
    trailing_edge_angle_deg: float
    n_points: int


@router.post("/thickness", response_model=ThicknessResponse)
def compute_thickness(req: ThicknessRequest) -> ThicknessResponse:
    """Compute blade thickness distribution.

    Supports 7 types: naca_4digit, elliptic, biparabolic, linear_taper,
    constant, dca, wedge.
    """
    # Backward compatibility: map legacy profile_type to thickness_type
    tt = req.thickness_type
    if req.profile_type is not None:
        if req.profile_type == 'naca':
            tt = 'naca_4digit'
        elif req.profile_type == 'ellipse':
            tt = 'elliptic'

    try:
        resolved_type = ThicknessType(tt)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail=f"Unknown thickness_type '{tt}'. "
                   f"Valid: {[t.value for t in ThicknessType]}",
        )

    r = get_thickness(
        thickness_type=resolved_type,
        t_max_frac=req.t_max_frac,
        n_points=req.n_points,
        close_te=req.close_te,
        le_ratio=req.le_ratio,
        te_ratio=req.te_ratio,
        s_max=req.s_max,
        t_min_frac=req.t_min_frac,
    )
    return ThicknessResponse(thickness_type=resolved_type.value, **r.__dict__)


# ── Stacking ──────────────────────────────────────────────────────────────────

class StackingRequest(BaseModel):
    d2: float = Field(..., gt=0)
    d1: float = Field(..., gt=0)
    blade_count: int = Field(..., ge=3, le=20)
    nq: float = Field(..., gt=0)
    lean_angle_deg: float = Field(0.0, ge=-30.0, le=30.0)
    sweep_mm: float = Field(0.0)
    bow_enabled: bool = False
    bow_angle_deg: float = Field(0.0)
    wrap_angle_deg: float = Field(70.0, ge=20.0, le=180.0)
    stack_reference: str = 'le'


class StackingResponse(BaseModel):
    wrap_angle_deg: float
    lean_angle_deg: float
    sweep_mm: float
    pitchwise_offset_hub: float
    pitchwise_offset_mid: float
    pitchwise_offset_shr: float
    axial_offset_hub_mm: float
    axial_offset_shr_mm: float
    lean_reduces_secondary_flow: bool
    warnings: list[str]


@router.post("/stacking", response_model=StackingResponse)
def compute_stacking_endpoint(req: StackingRequest) -> StackingResponse:
    """Compute blade stacking geometry (lean, bow, sweep, wrap angle)."""
    cfg = StackingConfig(
        lean_angle_deg=req.lean_angle_deg,
        sweep_mm=req.sweep_mm,
        bow_enabled=req.bow_enabled,
        bow_angle_deg=req.bow_angle_deg,
        wrap_angle_deg=req.wrap_angle_deg,
        stack_reference=req.stack_reference,
    )
    r = compute_stacking(cfg, req.d2, req.d1, req.blade_count, req.nq)
    return StackingResponse(**r.__dict__)


class WrapAngleRequest(BaseModel):
    d1: float = Field(..., gt=0)
    d2: float = Field(..., gt=0)
    beta1: float = Field(..., gt=0, lt=90)
    beta2: float = Field(..., gt=0, lt=90)
    blade_count: int = Field(..., ge=3, le=20)


@router.get("/wrap_angle")
def estimate_wrap_angle(d1: float, d2: float, beta1: float, beta2: float, blade_count: int) -> dict:
    """Estimate blade wrap angle from inlet/outlet angles (logarithmic spiral)."""
    wrap = wrap_angle_from_geometry(d1, d2, beta1, beta2, blade_count)
    return {"wrap_angle_deg": round(wrap, 1)}
