"""API routes for expanded turbomachine types and fluid properties.

Endpoints:
    POST /api/v1/sizing/radial_turbine — Radial inflow turbine sizing
    POST /api/v1/sizing/axial_fan      — Axial fan sizing
    POST /api/v1/sizing/sirocco_fan    — Forward-curved (sirocco) fan sizing
    POST /api/v1/fluids/properties     — Fluid property lookup
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ======================================================================
# Request / Response schemas
# ======================================================================


class GasPropsRequest(BaseModel):
    """Gas properties for the radial turbine."""

    name: str = Field("Air", description="Gas name")
    gamma: float = Field(1.4, gt=1.0, description="Cp/Cv ratio")
    R: float = Field(287.05, gt=0, description="Specific gas constant [J/(kg K)]")
    cp: float = Field(1004.5, gt=0, description="Cp [J/(kg K)]")


class RadialTurbineRequest(BaseModel):
    """Request body for radial inflow turbine sizing."""

    P_total_in: float = Field(..., gt=0, description="Inlet total pressure [Pa]")
    T_total_in: float = Field(..., gt=0, description="Inlet total temperature [K]")
    p_out: float = Field(..., gt=0, description="Outlet static pressure [Pa]")
    mass_flow: float = Field(..., gt=0, description="Mass flow rate [kg/s]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min]")
    gas_props: Optional[GasPropsRequest] = None


class RadialTurbineResponse(BaseModel):
    """Response body for radial inflow turbine sizing."""

    ns: float
    ns_dim: float
    d2: float
    d1: float
    b2: float
    alpha2: float
    beta2: float
    beta1: float
    u_c0: float
    c0: float
    u2: float
    cm2: float
    cm1: float
    pressure_ratio: float
    loss_nozzle: float
    loss_rotor: float
    loss_tip_clearance: float
    loss_exit_ke: float
    loss_total: float
    eta_ts: float
    eta_tt: float
    power: float
    mass_flow: float
    rpm: float
    blade_count: int
    warnings: List[str]


class AxialFanRequest(BaseModel):
    """Request body for axial fan sizing."""

    flow_rate: float = Field(..., gt=0, description="Volume flow rate Q [m^3/s]")
    total_pressure_rise: float = Field(..., gt=0, description="Total pressure rise [Pa]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min]")
    hub_tip_ratio: float = Field(0.5, ge=0.2, le=0.9, description="Hub-to-tip ratio")
    rho: float = Field(1.2, gt=0, description="Air density [kg/m^3]")


class AxialFanResponse(BaseModel):
    """Response body for axial fan sizing."""

    d_tip: float
    d_hub: float
    hub_tip_ratio: float
    blade_height: float
    d_mean: float
    blade_count: int
    chord: float
    solidity: float
    stagger_angle: float
    beta1_mean: float
    beta2_mean: float
    alpha1_mean: float
    alpha2_mean: float
    de_haller: float
    diffusion_factor: float
    fan_static_efficiency: float
    fan_total_efficiency: float
    power: float
    flow_rate: float
    total_pressure_rise: float
    static_pressure_rise: float
    axial_velocity: float
    tip_speed: float
    specific_speed: float
    flow_coefficient: float
    pressure_coefficient: float
    loss_profile: float
    loss_secondary: float
    loss_tip_clearance: float
    loss_annulus: float
    warnings: List[str]


class SiroccoFanRequest(BaseModel):
    """Request body for sirocco fan sizing."""

    flow_rate: float = Field(..., gt=0, description="Volume flow rate Q [m^3/s]")
    static_pressure: float = Field(..., gt=0, description="Static pressure rise [Pa]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min]")
    rho: float = Field(1.2, gt=0, description="Air density [kg/m^3]")


class SiroccoFanResponse(BaseModel):
    """Response body for sirocco fan sizing."""

    d2: float
    d1: float
    b2: float
    d1_d2: float
    blade_count: int
    beta2: float
    beta1: float
    blade_chord: float
    scroll_width: float
    scroll_d_outer: float
    cutoff_clearance: float
    flow_rate: float
    static_pressure: float
    total_pressure: float
    power: float
    fan_static_efficiency: float
    fan_total_efficiency: float
    tip_speed: float
    rpm: float
    specific_speed: float
    flow_coefficient: float
    pressure_coefficient: float
    cm2: float
    cu2: float
    c2: float
    w2: float
    warnings: List[str]


class FluidPropertiesRequest(BaseModel):
    """Request body for fluid property lookup."""

    fluid: str = Field(..., description="Fluid name (e.g. water, air, R134a, CO2)")
    T: Optional[float] = Field(None, gt=0, description="Temperature [K] (for real-gas lookup)")
    p: Optional[float] = Field(None, gt=0, description="Pressure [Pa] (for real-gas lookup)")


class FluidPropertiesResponse(BaseModel):
    """Response body for fluid properties."""

    name: str
    rho: float
    mu: float
    gamma: float
    cp: float
    R_specific: float
    p_vapor: float
    is_compressible: bool
    compressibility_factor: Optional[float] = None


# ======================================================================
# Router
# ======================================================================

router = APIRouter(prefix="/api/v1", tags=["turbotypes", "fluids"])


@router.post("/sizing/radial_turbine", response_model=RadialTurbineResponse)
def sizing_radial_turbine(req: RadialTurbineRequest) -> RadialTurbineResponse:
    """Preliminary sizing of a radial inflow turbine."""
    from hpe.sizing.radial_inflow_turbine import size_radial_turbine, GasProps

    gas = None
    if req.gas_props is not None:
        gas = GasProps(
            name=req.gas_props.name,
            gamma=req.gas_props.gamma,
            R=req.gas_props.R,
            cp=req.gas_props.cp,
        )

    try:
        result = size_radial_turbine(
            P_total_in=req.P_total_in,
            T_total_in=req.T_total_in,
            p_out=req.p_out,
            mass_flow=req.mass_flow,
            rpm=req.rpm,
            gas_props=gas,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RadialTurbineResponse(
        ns=result.ns,
        ns_dim=result.ns_dim,
        d2=result.d2,
        d1=result.d1,
        b2=result.b2,
        alpha2=result.alpha2,
        beta2=result.beta2,
        beta1=result.beta1,
        u_c0=result.u_c0,
        c0=result.c0,
        u2=result.u2,
        cm2=result.cm2,
        cm1=result.cm1,
        pressure_ratio=result.pressure_ratio,
        loss_nozzle=result.loss_nozzle,
        loss_rotor=result.loss_rotor,
        loss_tip_clearance=result.loss_tip_clearance,
        loss_exit_ke=result.loss_exit_ke,
        loss_total=result.loss_total,
        eta_ts=result.eta_ts,
        eta_tt=result.eta_tt,
        power=result.power,
        mass_flow=result.mass_flow,
        rpm=result.rpm,
        blade_count=result.blade_count,
        warnings=result.warnings,
    )


@router.post("/sizing/axial_fan", response_model=AxialFanResponse)
def sizing_axial_fan(req: AxialFanRequest) -> AxialFanResponse:
    """Preliminary sizing of an axial fan."""
    from hpe.sizing.axial_fan import size_axial_fan

    try:
        result = size_axial_fan(
            flow_rate=req.flow_rate,
            total_pressure_rise=req.total_pressure_rise,
            rpm=req.rpm,
            hub_tip_ratio=req.hub_tip_ratio,
            rho=req.rho,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AxialFanResponse(
        d_tip=result.d_tip,
        d_hub=result.d_hub,
        hub_tip_ratio=result.hub_tip_ratio,
        blade_height=result.blade_height,
        d_mean=result.d_mean,
        blade_count=result.blade_count,
        chord=result.chord,
        solidity=result.solidity,
        stagger_angle=result.stagger_angle,
        beta1_mean=result.beta1_mean,
        beta2_mean=result.beta2_mean,
        alpha1_mean=result.alpha1_mean,
        alpha2_mean=result.alpha2_mean,
        de_haller=result.de_haller,
        diffusion_factor=result.diffusion_factor,
        fan_static_efficiency=result.fan_static_efficiency,
        fan_total_efficiency=result.fan_total_efficiency,
        power=result.power,
        flow_rate=result.flow_rate,
        total_pressure_rise=result.total_pressure_rise,
        static_pressure_rise=result.static_pressure_rise,
        axial_velocity=result.axial_velocity,
        tip_speed=result.tip_speed,
        specific_speed=result.specific_speed,
        flow_coefficient=result.flow_coefficient,
        pressure_coefficient=result.pressure_coefficient,
        loss_profile=result.loss_profile,
        loss_secondary=result.loss_secondary,
        loss_tip_clearance=result.loss_tip_clearance,
        loss_annulus=result.loss_annulus,
        warnings=result.warnings,
    )


@router.post("/sizing/sirocco_fan", response_model=SiroccoFanResponse)
def sizing_sirocco_fan(req: SiroccoFanRequest) -> SiroccoFanResponse:
    """Preliminary sizing of a forward-curved (sirocco) fan."""
    from hpe.sizing.sirocco_fan import size_sirocco_fan

    try:
        result = size_sirocco_fan(
            flow_rate=req.flow_rate,
            static_pressure=req.static_pressure,
            rpm=req.rpm,
            rho=req.rho,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SiroccoFanResponse(
        d2=result.d2,
        d1=result.d1,
        b2=result.b2,
        d1_d2=result.d1_d2,
        blade_count=result.blade_count,
        beta2=result.beta2,
        beta1=result.beta1,
        blade_chord=result.blade_chord,
        scroll_width=result.scroll_width,
        scroll_d_outer=result.scroll_d_outer,
        cutoff_clearance=result.cutoff_clearance,
        flow_rate=result.flow_rate,
        static_pressure=result.static_pressure,
        total_pressure=result.total_pressure,
        power=result.power,
        fan_static_efficiency=result.fan_static_efficiency,
        fan_total_efficiency=result.fan_total_efficiency,
        tip_speed=result.tip_speed,
        rpm=result.rpm,
        specific_speed=result.specific_speed,
        flow_coefficient=result.flow_coefficient,
        pressure_coefficient=result.pressure_coefficient,
        cm2=result.cm2,
        cu2=result.cu2,
        c2=result.c2,
        w2=result.w2,
        warnings=result.warnings,
    )


@router.post("/fluids/properties", response_model=FluidPropertiesResponse)
def get_fluid_properties(req: FluidPropertiesRequest) -> FluidPropertiesResponse:
    """Look up fluid properties, optionally at a given T and p.

    If T and p are provided, uses real-gas tables for refrigerants/gases.
    Otherwise returns the predefined reference-state properties.
    """
    from hpe.physics.fluid_properties import get_fluid, FluidProperties

    try:
        if req.T is not None and req.p is not None:
            # Real-gas lookup
            props = FluidProperties.from_real_gas_table(req.fluid, req.T, req.p)
            Z = props.compressibility_factor(req.T, req.p)
        else:
            props = get_fluid(req.fluid)
            Z = None
    except (KeyError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FluidPropertiesResponse(
        name=props.name,
        rho=props.rho,
        mu=props.mu,
        gamma=props.gamma,
        cp=props.cp,
        R_specific=props.R_specific,
        p_vapor=props.p_vapor,
        is_compressible=props.is_compressible(),
        compressibility_factor=Z,
    )
