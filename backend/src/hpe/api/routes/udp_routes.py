"""User Defined Parameters (UDP) API routes.

Endpoints:
    GET  /api/v1/udp/list      — list all registered UDPs
    POST /api/v1/udp/evaluate  — compute all UDPs for a design
    POST /api/v1/udp/register  — register a custom UDP via expression string
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["udp"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class UDPParamInfo(BaseModel):
    name: str
    description: str
    unit: str
    category: str


class UDPListResponse(BaseModel):
    parameters: List[UDPParamInfo]
    count: int


class UDPEvaluateRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Volume flow rate Q [m^3/s]")
    head: float = Field(..., gt=0, description="Total head H [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min]")
    machine_type: str = Field("centrifugal_pump", description="Machine type")


class UDPValueOut(BaseModel):
    value: float
    unit: str
    description: str


class UDPEvaluateResponse(BaseModel):
    parameters: Dict[str, UDPValueOut]


class UDPRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80, description="Parameter name")
    expression: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Python math expression using sizing result fields. "
            "Available variables: d1, d2, b2, beta1, beta2, blade_count, "
            "eta, power, npsh_r, ns, nq, Q, H, n, rho. "
            "Available functions: math.* and numpy basics."
        ),
    )
    unit: str = Field("-", description="Unit string")
    description: str = Field("Custom UDP", description="Parameter description")


class UDPRegisterResponse(BaseModel):
    name: str
    status: str


# ---------------------------------------------------------------------------
# Safe eval namespace
# ---------------------------------------------------------------------------

import numpy as _np

_SAFE_NAMESPACE: Dict[str, Any] = {
    # math functions
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "exp": math.exp,
    "pow": pow,
    "abs": abs,
    "min": min,
    "max": max,
    "radians": math.radians,
    "degrees": math.degrees,
    # numpy basics
    "np_sqrt": _np.sqrt,
    "np_mean": _np.mean,
    "np_sum": _np.sum,
}


def _build_expression_fn(expression: str, name: str):
    """Build a safe compute function from a user expression string."""
    # Compile once to catch syntax errors early
    try:
        code = compile(expression, f"<udp:{name}>", "eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # Block dangerous builtins
    for forbidden in ("import", "__", "exec", "eval", "open", "compile", "globals", "locals"):
        if forbidden in expression:
            raise ValueError(f"Expression contains forbidden token: '{forbidden}'")

    def compute_fn(sr, op):
        from hpe.core.models import G
        ns = {
            **_SAFE_NAMESPACE,
            # Sizing result fields
            "d1": sr.impeller_d1,
            "d2": sr.impeller_d2,
            "b2": sr.impeller_b2,
            "beta1": sr.beta1,
            "beta2": sr.beta2,
            "blade_count": sr.blade_count,
            "eta": sr.estimated_efficiency,
            "power": sr.estimated_power,
            "npsh_r": sr.estimated_npsh_r,
            "ns": sr.specific_speed_ns,
            "nq": sr.specific_speed_nq,
            # Operating point fields
            "Q": op.flow_rate,
            "H": op.head,
            "n": op.rpm,
            "rho": op.fluid_density,
            "g": G,
        }
        return float(eval(code, {"__builtins__": {}}, ns))  # noqa: S307

    return compute_fn


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/udp/list", response_model=UDPListResponse)
def list_udps() -> UDPListResponse:
    """List all registered UDPs."""
    from hpe.core.udp import get_registry

    reg = get_registry()
    params = reg.list_parameters()
    return UDPListResponse(
        parameters=[UDPParamInfo(**p) for p in params],
        count=len(params),
    )


@router.post("/udp/evaluate", response_model=UDPEvaluateResponse)
def evaluate_udps(req: UDPEvaluateRequest) -> UDPEvaluateResponse:
    """Compute all UDPs for a given design (runs sizing internally)."""
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint
    from hpe.core.udp import get_registry
    from hpe.sizing import run_sizing

    op = OperatingPoint(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        machine_type=MachineType(req.machine_type),
    )

    try:
        sr = run_sizing(op)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Sizing failed: {exc}")

    reg = get_registry()
    results = reg.evaluate_all(sr, op)

    return UDPEvaluateResponse(
        parameters={
            name: UDPValueOut(value=r.value, unit=r.unit, description=r.description)
            for name, r in results.items()
        }
    )


@router.post("/udp/register", response_model=UDPRegisterResponse)
def register_udp(req: UDPRegisterRequest) -> UDPRegisterResponse:
    """Register a custom UDP via a safe math expression."""
    from hpe.core.udp import get_registry

    try:
        fn = _build_expression_fn(req.expression, req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reg = get_registry()
    reg.register(
        name=req.name,
        compute_fn=fn,
        description=req.description,
        unit=req.unit,
        category="custom",
    )
    return UDPRegisterResponse(name=req.name, status="registered")
