"""Endpoints de cálculos físicos clássicos — melhorias #11-20.

Todos sob /api/v1/physics/*
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/physics", tags=["physics"])


# ===========================================================================
# #16 Slip factor (3 modelos)
# ===========================================================================

class SlipFactorRequest(BaseModel):
    n_blades: int = Field(..., ge=2, le=20)
    beta2_deg: float = Field(..., gt=0, lt=90)
    d1_d2_ratio: float = Field(0.5, gt=0.1, lt=1.0)


@router.post("/slip_factor", summary="Slip factor — Wiesner/Stodola/Stanitz")
def slip_factor(req: SlipFactorRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_slip_factors
    return compute_slip_factors(req.n_blades, req.beta2_deg, req.d1_d2_ratio).to_dict()


# ===========================================================================
# #17 Affinity laws scaling
# ===========================================================================

class AffinityRequest(BaseModel):
    Q_old: float = Field(..., gt=0)
    H_old: float = Field(..., gt=0)
    P_old: float = Field(..., gt=0)
    eta_old: float = Field(..., gt=0, lt=1)
    n_old: float = Field(..., gt=0)
    n_new: float = Field(..., gt=0)
    d_old: float = 1.0
    d_new: float = 1.0
    apply_re_correction: bool = True


@router.post("/affinity_scaling", summary="Affinity laws scaling Q-H-P-η")
def affinity_scaling(req: AffinityRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import apply_affinity_laws
    return apply_affinity_laws(
        req.Q_old, req.H_old, req.P_old, req.eta_old,
        req.n_old, req.n_new, req.d_old, req.d_new,
        apply_re_correction=req.apply_re_correction,
    ).to_dict()


# ===========================================================================
# #18 Disk friction
# ===========================================================================

class DiskFrictionRequest(BaseModel):
    d2: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    rho: float = 998.2
    nu: float = 1e-6
    s_axial_gap: float = 0.005


@router.post("/disk_friction", summary="Disk friction loss (Daily-Nece)")
def disk_friction(req: DiskFrictionRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_disk_friction
    return compute_disk_friction(
        req.d2, req.rpm, req.rho, req.nu, req.s_axial_gap,
    ).to_dict()


# ===========================================================================
# #19 Volumetric efficiency
# ===========================================================================

class VolumetricRequest(BaseModel):
    Q: float = Field(..., gt=0)
    H: float = Field(..., gt=0)
    d_seal: float = Field(..., gt=0)
    clearance: float = 0.0003
    seal_length: float = 0.020


@router.post("/volumetric_efficiency", summary="η volumetric (clearance leakage)")
def volumetric_efficiency(req: VolumetricRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_volumetric_efficiency
    return compute_volumetric_efficiency(
        req.Q, req.H, req.d_seal, req.clearance, req.seal_length,
    ).to_dict()


# ===========================================================================
# #20 Reynolds correction (η scaling)
# ===========================================================================

class ReynoldsRequest(BaseModel):
    eta_ref: float = Field(..., gt=0, lt=1)
    Re_ref: float = Field(..., gt=0)
    Re_target: float = Field(..., gt=0)
    method: str = Field("moody", description="moody | ackeret | pfleider")


@router.post("/reynolds_correction", summary="Reynolds η scaling (Moody/Ackeret)")
def reynolds_correction(req: ReynoldsRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_reynolds_correction
    eta_new = compute_reynolds_correction(
        req.eta_ref, req.Re_ref, req.Re_target, method=req.method,
    )
    return {
        "eta_old": req.eta_ref,
        "eta_new": round(eta_new, 4),
        "delta_eta": round(eta_new - req.eta_ref, 4),
        "method": req.method,
    }


# ===========================================================================
# Mechanical efficiency
# ===========================================================================

class MechanicalRequest(BaseModel):
    P_hydraulic: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    n_bearings: int = 2
    n_seals: int = 1


@router.post("/mechanical_efficiency", summary="η mechanical (bearing+seal)")
def mechanical_efficiency(req: MechanicalRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_mechanical_efficiency
    return compute_mechanical_efficiency(
        req.P_hydraulic, req.rpm, req.n_bearings, n_seals=req.n_seals,
    ).to_dict()


# ===========================================================================
# Suction specific speed Nss + specific diameter Ds
# ===========================================================================

class NssRequest(BaseModel):
    Q: float = Field(..., gt=0)
    npsh_r: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)


@router.post("/nss", summary="Suction specific speed Nss")
def nss_endpoint(req: NssRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import compute_suction_specific_speed
    nss = compute_suction_specific_speed(req.Q, req.npsh_r, req.rpm)
    safety = "conservative" if nss < 8000 else "industry" if nss < 12000 else "aggressive"
    return {"nss": round(nss, 0), "safety": safety}


class DsRequest(BaseModel):
    d2: float = Field(..., gt=0)
    H: float = Field(..., gt=0)
    Q: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)


@router.post("/specific_diameter", summary="Cordier specific diameter Ds + ωs")
def specific_diameter(req: DsRequest) -> dict[str, Any]:
    from hpe.physics.loss_correlations import (
        compute_specific_diameter, compute_specific_speed_omega,
    )
    Ds = compute_specific_diameter(req.d2, req.H, req.Q)
    omega_s = compute_specific_speed_omega(req.Q, req.H, req.rpm)
    return {
        "Ds": round(Ds, 3),
        "omega_s": round(omega_s, 4),
        "cordier_optimal": 0.5 < omega_s < 1.5,   # Cordier optimal range
    }
