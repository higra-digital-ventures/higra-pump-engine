"""Convergence solver API routes — iterative blade design.

Endpoints:
    POST /api/v1/design/converge  — run full convergence, return history + blade shape
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["convergence"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ConvergeRequest(BaseModel):
    """Request body for the convergence solver endpoint."""

    # Operating conditions
    flow_rate: float = Field(..., gt=0, description="Volume flow rate Q [m^3/s].")
    head: float = Field(..., gt=0, description="Design head H [m].")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min].")

    # Geometry from 1D sizing
    r1: float = Field(..., gt=0, description="Inlet radius [m].")
    r2: float = Field(..., gt=0, description="Outlet radius [m].")
    b1: float = Field(..., gt=0, description="Inlet width [m].")
    b2: float = Field(..., gt=0, description="Outlet width [m].")
    blade_count: int = Field(..., ge=3, le=30, description="Number of blades Z.")
    beta1: float = Field(..., ge=5, le=85, description="Initial inlet blade angle [deg].")
    beta2: float = Field(..., ge=5, le=85, description="Initial outlet blade angle [deg].")

    # Loading control
    loading_type: str = Field(
        "mid_loaded",
        description="Loading template: front_loaded, mid_loaded, aft_loaded, controlled_diffusion.",
    )

    # Solver parameters
    n_chord: int = Field(51, ge=20, le=200, description="Streamwise stations.")
    n_span: int = Field(5, ge=3, le=11, description="Spanwise stations.")
    max_iterations: int = Field(50, ge=5, le=200, description="Max iterations.")
    damping_factor: float = Field(0.5, ge=0.1, le=0.9, description="Under-relaxation factor.")
    slip_model: str = Field("wiesner", description="Slip model: wiesner or stodola.")


class IterationHistoryItem(BaseModel):
    iteration: int
    max_residual_deg: float
    wrap_angle_hub_deg: float
    wrap_angle_shroud_deg: float
    beta1_deg: float
    beta2_deg: float


class ConvergeResponse(BaseModel):
    """Response body for the convergence solver endpoint."""

    converged: bool
    iterations: int
    max_residual_deg: float
    history: List[IterationHistoryItem]
    blade_angles_inlet: List[float]
    blade_angles_outlet: List[float]
    wrap_angles: List[float]
    beta_distribution: List[List[float]]
    wrap_angle_distribution: List[List[float]]
    loading_warnings: List[str]
    loading_errors: List[str]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/design/converge", response_model=ConvergeResponse)
def run_convergence_endpoint(req: ConvergeRequest) -> ConvergeResponse:
    """Run the iterative convergence solver.

    Uses the 1D sizing result (r1, r2, b1, b2, beta1, beta2) as the
    initial guess and iterates with the prescribed loading distribution
    until blade angles converge.
    """
    from hpe.sizing.convergence_solver import run_convergence

    valid_types = {"front_loaded", "mid_loaded", "aft_loaded", "controlled_diffusion"}
    if req.loading_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid loading_type '{req.loading_type}'. "
                   f"Must be one of: {', '.join(sorted(valid_types))}.",
        )

    if req.r2 <= req.r1:
        raise HTTPException(
            status_code=422,
            detail="Outlet radius r2 must be greater than inlet radius r1.",
        )

    try:
        result = run_convergence(
            flow_rate=req.flow_rate,
            head=req.head,
            rpm=req.rpm,
            r1=req.r1,
            r2=req.r2,
            b1=req.b1,
            b2=req.b2,
            blade_count=req.blade_count,
            beta1=req.beta1,
            beta2=req.beta2,
            loading_type=req.loading_type,
            n_chord=req.n_chord,
            n_span=req.n_span,
            max_iterations=req.max_iterations,
            damping_factor=req.damping_factor,
            slip_model=req.slip_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data = result.to_dict()
    return ConvergeResponse(**data)
