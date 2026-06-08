"""Analysis API routes — losses, stress, inverse design."""

from __future__ import annotations


from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List

from hpe.core.models import OperatingPoint

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalysisRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)


class LossResponse(BaseModel):
    profile_loss_ps: float
    profile_loss_ss: float
    profile_loss_total: float
    tip_leakage: float
    endwall_hub: float
    endwall_shroud: float
    endwall_total: float
    mixing: float
    incidence: float
    disk_friction_power: float
    recirculation: float
    total_head_loss: float
    loss_coefficient: float


class StressResponse(BaseModel):
    centrifugal_stress_root: float
    centrifugal_stress_tip: float
    bending_stress_le: float
    bending_stress_te: float
    bending_stress_max: float
    von_mises_max: float
    sf_yield: float
    sf_fatigue: float
    sf_ultimate: float
    first_natural_freq: float
    campbell_margin: float
    is_safe: bool
    warnings: List[str]


class InverseRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    loading_type: str = Field("mid_loaded")
    n_spans: int = Field(5, ge=3, le=11)


class InverseResponse(BaseModel):
    beta_inlet: List[float]
    beta_outlet: List[float]
    wrap_angles: List[float]
    span_fractions: List[float]
    max_blade_loading: float
    diffusion_ratio: float


@router.post("/losses", response_model=LossResponse)
def loss_breakdown_endpoint(req: AnalysisRequest) -> LossResponse:
    """Calculate advanced loss breakdown."""
    from hpe.physics.advanced_losses import calc_advanced_losses
    from hpe.physics.euler import calc_off_design_triangles, get_design_flow_rate
    from hpe.sizing import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)
    q_design = get_design_flow_rate(sizing)
    tri_in, tri_out = calc_off_design_triangles(sizing, req.flow_rate)

    result = calc_advanced_losses(
        sizing, q_actual=req.flow_rate, q_design=q_design,
        tri_in=tri_in, tri_out=tri_out,
    )

    return LossResponse(
        profile_loss_ps=result.profile_loss_ps,
        profile_loss_ss=result.profile_loss_ss,
        profile_loss_total=result.profile_loss_total,
        tip_leakage=result.tip_leakage,
        endwall_hub=result.endwall_hub,
        endwall_shroud=result.endwall_shroud,
        endwall_total=result.endwall_total,
        mixing=result.mixing,
        incidence=result.incidence,
        disk_friction_power=result.disk_friction_power,
        recirculation=result.recirculation,
        total_head_loss=result.total_head_loss,
        loss_coefficient=result.loss_coefficient,
    )


@router.post("/stress", response_model=StressResponse)
def stress_endpoint(req: AnalysisRequest) -> StressResponse:
    """Run blade stress analysis."""
    from hpe.physics.stress import analyze_stress
    from hpe.sizing import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    result = analyze_stress(
        sizing, rpm=req.rpm, head=req.head, flow_rate=req.flow_rate,
    )

    return StressResponse(
        centrifugal_stress_root=result.centrifugal_stress_root,
        centrifugal_stress_tip=result.centrifugal_stress_tip,
        bending_stress_le=result.bending_stress_le,
        bending_stress_te=result.bending_stress_te,
        bending_stress_max=result.bending_stress_max,
        von_mises_max=result.von_mises_max,
        sf_yield=result.sf_yield,
        sf_fatigue=result.sf_fatigue,
        sf_ultimate=result.sf_ultimate,
        first_natural_freq=result.first_natural_freq,
        campbell_margin=result.campbell_margin,
        is_safe=result.is_safe,
        warnings=result.warnings,
    )


@router.post("/inverse", response_model=InverseResponse)
def inverse_design_endpoint(req: InverseRequest) -> InverseResponse:
    """Run inverse blade design from loading specification."""
    from hpe.geometry.inverse.models import InverseDesignSpec, LoadingType
    from hpe.geometry.inverse.solver import inverse_design
    from hpe.sizing import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    loading_type = LoadingType(req.loading_type)
    spec = InverseDesignSpec.from_sizing_result(sizing, rpm=req.rpm, loading_type=loading_type)
    spec.n_spanwise = req.n_spans

    result = inverse_design(spec)

    return InverseResponse(
        beta_inlet=result.beta_inlet,
        beta_outlet=result.beta_outlet,
        wrap_angles=result.wrap_angles,
        span_fractions=result.span_fractions,
        max_blade_loading=result.max_blade_loading,
        diffusion_ratio=result.diffusion_ratio,
    )
