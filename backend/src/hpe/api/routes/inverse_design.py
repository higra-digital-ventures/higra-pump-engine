"""Inverse design API routes — blade loading and meridional SLC solver.

Endpoints:
    POST /api/v1/geometry/loading/distribution  — rVθ* S-curve loading (B1)
    POST /api/v1/geometry/loading/pressure      — PS/SS pressure distribution (B3)
    POST /api/v1/geometry/meridional/slc        — Streamline curvature solver (B2)
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["inverse-design"])


# ---------------------------------------------------------------------------
# Pydantic models — Loading Distribution (B1)
# ---------------------------------------------------------------------------

class LoadingDistributionParams(BaseModel):
    """S-curve loading parameters for one span station."""

    nc: float = Field(0.20, ge=0.0, le=1.0, description="Normalized position of max loading gradient (LE side).")
    nd: float = Field(0.80, ge=0.0, le=1.0, description="Normalized position of plateau end (TE side).")
    slope: float = Field(1.5, description="TE slope: positive = forward loaded, negative = aft loaded.")
    drvt_le: float = Field(0.0, description="Derivative of rVθ* at LE (0 = no LE spike).")
    rvt_te: float = Field(0.523, gt=0.0, description="Target rVθ* at TE [m²/s] (from Euler equation).")
    n_points: int = Field(51, ge=10, le=201, description="Number of chord discretisation points.")


class LoadingDistributionRequest(BaseModel):
    hub: LoadingDistributionParams = Field(default_factory=LoadingDistributionParams)
    shroud: LoadingDistributionParams = Field(default_factory=LoadingDistributionParams)
    n_span: int = Field(3, ge=2, le=10, description="Number of spanwise stations.")


class SpanLoadingResult(BaseModel):
    m_norm: List[float]
    rvt: List[float]
    drvt_dm: List[float]
    ps_excess: List[float]
    ss_excess: List[float]


class LoadingDistributionResponse(BaseModel):
    hub: SpanLoadingResult
    mid: SpanLoadingResult
    shroud: SpanLoadingResult


# ---------------------------------------------------------------------------
# Pydantic models — Blade Pressure Distribution (B3)
# ---------------------------------------------------------------------------

class BladePressureRequest(BaseModel):
    """Request for PS/SS pressure distribution from a single span loading."""

    m_norm: List[float] = Field(..., min_length=2, description="Normalized chord positions [0..1].")
    rvt: List[float] = Field(..., min_length=2, description="rVθ* distribution [m²/s].")
    drvt_dm: List[float] = Field(..., min_length=2, description="d(rVθ*)/dm loading rate.")
    ps_excess: List[float] = Field(..., min_length=2)
    ss_excess: List[float] = Field(..., min_length=2)
    w_inlet: float = Field(..., gt=0.0, description="Inlet relative velocity [m/s].")
    w_outlet: float = Field(..., gt=0.0, description="Outlet relative velocity [m/s].")
    rho: float = Field(998.0, gt=0.0, description="Fluid density [kg/m³].")


class BladePressureResponse(BaseModel):
    m_norm: List[float]
    w_ps: List[float]
    w_ss: List[float]
    cp_ps: List[float]
    cp_ss: List[float]
    w_ref: float


# ---------------------------------------------------------------------------
# Pydantic models — Meridional SLC (B2)
# ---------------------------------------------------------------------------

class SLCRequest(BaseModel):
    flow_rate: float = Field(..., gt=0.0, description="Volumetric flow rate Q [m³/s].")
    rpm: float = Field(..., gt=0.0, description="Rotational speed [rev/min].")
    hub_profile_r: List[float] = Field(..., min_length=2, description="Hub radial coordinates from LE to TE [m].")
    hub_profile_z: List[float] = Field(..., min_length=2, description="Hub axial coordinates from LE to TE [m].")
    shr_profile_r: List[float] = Field(..., min_length=2, description="Shroud radial coordinates from LE to TE [m].")
    shr_profile_z: List[float] = Field(..., min_length=2, description="Shroud axial coordinates from LE to TE [m].")
    n_stations: int = Field(5, ge=2, le=50, description="Number of radial-equilibrium computing stations.")
    max_iter: int = Field(30, ge=1, le=200)
    tol: float = Field(1e-4, gt=0.0)


class SLCResponse(BaseModel):
    r_stations: List[float]
    z_stations: List[float]
    cm_meridional: List[float]
    cu_swirl: List[float]
    pressure: List[float]
    streamline_curvature: List[float]
    is_converged: bool
    iterations: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/geometry/loading/distribution", response_model=LoadingDistributionResponse)
def loading_distribution(req: LoadingDistributionRequest) -> LoadingDistributionResponse:
    """Calculate rVθ* S-curve loading distribution from hub to shroud.

    Returns hub, mid, and shroud BladeLoadingResult objects containing the
    rVθ* distribution along the normalized chord and the PS/SS excess
    velocities derived from the loading rate.
    """
    from hpe.geometry.inverse.loading import (
        LoadingDistribution,
        calc_loading_distribution,
    )

    hub_ld = LoadingDistribution(
        nc=req.hub.nc,
        nd=req.hub.nd,
        slope=req.hub.slope,
        drvt_le=req.hub.drvt_le,
        rvt_te=req.hub.rvt_te,
        n_points=req.hub.n_points,
    )
    shroud_ld = LoadingDistribution(
        nc=req.shroud.nc,
        nd=req.shroud.nd,
        slope=req.shroud.slope,
        drvt_le=req.shroud.drvt_le,
        rvt_te=req.shroud.rvt_te,
        n_points=req.shroud.n_points,
    )

    results = calc_loading_distribution(hub_ld, shroud_ld, n_span=req.n_span)

    def _to_response(r) -> SpanLoadingResult:
        return SpanLoadingResult(
            m_norm=r.m_norm,
            rvt=r.rvt,
            drvt_dm=r.drvt_dm,
            ps_excess=r.ps_excess,
            ss_excess=r.ss_excess,
        )

    return LoadingDistributionResponse(
        hub=_to_response(results["hub"]),
        mid=_to_response(results["mid"]),
        shroud=_to_response(results["shroud"]),
    )


@router.post("/geometry/loading/pressure", response_model=BladePressureResponse)
def blade_pressure_distribution(req: BladePressureRequest) -> BladePressureResponse:
    """Calculate PS and SS velocity/pressure distributions from blade loading.

    Takes a BladeLoadingResult (as flat fields) plus inlet/outlet relative
    velocities and returns Cp distributions on the pressure and suction sides.
    """
    from hpe.geometry.inverse.loading import (
        BladeLoadingResult,
        calc_blade_pressure_distribution,
    )

    lengths = {
        len(req.m_norm), len(req.rvt), len(req.drvt_dm),
        len(req.ps_excess), len(req.ss_excess),
    }
    if len(lengths) != 1:
        raise HTTPException(
            status_code=422,
            detail="m_norm, rvt, drvt_dm, ps_excess, ss_excess must all have the same length.",
        )

    loading = BladeLoadingResult(
        m_norm=req.m_norm,
        rvt=req.rvt,
        drvt_dm=req.drvt_dm,
        ps_excess=req.ps_excess,
        ss_excess=req.ss_excess,
    )

    result = calc_blade_pressure_distribution(
        loading=loading,
        w_inlet=req.w_inlet,
        w_outlet=req.w_outlet,
        rho=req.rho,
    )

    return BladePressureResponse(**result)


@router.post("/geometry/meridional/slc", response_model=SLCResponse)
def meridional_slc(req: SLCRequest) -> SLCResponse:
    """Solve meridional velocity distribution using the simplified SLC method.

    Given hub and shroud (r, z) profiles and operating conditions, computes
    the meridional velocity, swirl velocity, static pressure, and streamline
    curvature at each computing station from LE to TE.
    """
    from hpe.physics.slc import solve_meridional_slc

    if len(req.hub_profile_r) != len(req.hub_profile_z):
        raise HTTPException(
            status_code=422,
            detail="hub_profile_r and hub_profile_z must have the same length.",
        )
    if len(req.shr_profile_r) != len(req.shr_profile_z):
        raise HTTPException(
            status_code=422,
            detail="shr_profile_r and shr_profile_z must have the same length.",
        )

    result = solve_meridional_slc(
        flow_rate=req.flow_rate,
        rpm=req.rpm,
        hub_profile_r=req.hub_profile_r,
        hub_profile_z=req.hub_profile_z,
        shr_profile_r=req.shr_profile_r,
        shr_profile_z=req.shr_profile_z,
        n_stations=req.n_stations,
        max_iter=req.max_iter,
        tol=req.tol,
    )

    return SLCResponse(
        r_stations=result.r_stations,
        z_stations=result.z_stations,
        cm_meridional=result.cm_meridional,
        cu_swirl=result.cu_swirl,
        pressure=result.pressure,
        streamline_curvature=result.streamline_curvature,
        is_converged=result.is_converged,
        iterations=result.iterations,
    )
