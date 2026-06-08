"""Design database API endpoints.

    GET  /api/v1/design/machine_types              — list all machine types
    POST /api/v1/design/recommend                  — get preliminary design recommendation
    GET  /api/v1/design/cordier?nq=<nq>            — Cordier diagram point (sigma vs phi)
"""
from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from hpe.sizing.design_db import get_design_recommendation, list_machine_types

router = APIRouter(prefix="/api/v1/design", tags=["design-db"])


class RecommendRequest(BaseModel):
    machine_type: str
    nq: float = Field(..., gt=0, description="Specific speed Nq = N·√Q / H^0.75")
    blade_count_override: Optional[int] = None


class RecommendResponse(BaseModel):
    machine_type: str
    nq: float
    beta2_recommended_deg: float
    blade_count_recommended: int
    b2_d2_recommended: float
    phi_ref: float
    psi_ref: float
    eta_expected: float
    splitter_recommended: bool
    nq_assessment: str
    nq_distance_from_opt: float
    warnings: list[str]
    notes: str


@router.get("/machine_types")
def get_machine_types() -> list[dict]:
    """List all supported machine types with Nq ranges."""
    return list_machine_types()


@router.post("/recommend", response_model=RecommendResponse)
def get_recommendation(req: RecommendRequest) -> RecommendResponse:
    """Get preliminary design recommendation for a machine type at given Nq."""
    r = get_design_recommendation(req.machine_type, req.nq, req.blade_count_override)
    return RecommendResponse(**r.__dict__)


@router.get("/cordier")
def cordier_point(nq: float) -> dict:
    """Return Cordier diagram optimum (sigma, phi) for given Nq.

    Cordier (1953) diagram: optimum machines lie on the Cordier line.
    sigma = D * sqrt(rho * n²) / (2 * delta_p)^0.5  — shape number
    phi = Q / (n * D³)                               — flow coefficient

    Empirical Cordier correlations (Eck, 1973):
        sigma_opt = 0.449 * Nq^0.716  (with Nq in SI units, Nq = n*Q^0.5/H^0.75)
        phi_opt   = 0.0254 * Nq^1.34
    """
    sigma_opt = 0.449 * (nq ** 0.716)
    phi_opt = 0.0254 * (nq ** 1.34)

    # Machine type on Cordier line
    if nq < 15:
        machine_class = 'centrifugal (radial)'
    elif nq < 50:
        machine_class = 'centrifugal'
    elif nq < 120:
        machine_class = 'mixed-flow'
    else:
        machine_class = 'axial'

    return {
        'nq': nq,
        'sigma_optimal': round(sigma_opt, 4),
        'phi_optimal': round(phi_opt, 4),
        'machine_class': machine_class,
    }
