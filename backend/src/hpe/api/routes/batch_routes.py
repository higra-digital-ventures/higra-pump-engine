"""Batch / parametric automation API routes.

Endpoints:
    POST /api/v1/batch/parametric         — parametric sweep over one variable
    POST /api/v1/batch/multi_run          — multiple arbitrary sizing jobs
    POST /api/v1/batch/sensitivity_matrix — full Jacobian sensitivity matrix
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["batch"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ParametricRequest(BaseModel):
    """Parametric sweep input."""

    flow_rate: float = Field(..., gt=0, description="Base Q [m³/s]")
    head: float = Field(..., gt=0, description="Base H [m]")
    rpm: float = Field(..., gt=0, description="Base RPM")
    sweep_variable: str = Field(
        ..., description="Variable to sweep: 'flow_rate' | 'head' | 'rpm'"
    )
    sweep_min: float = Field(..., description="Sweep range minimum")
    sweep_max: float = Field(..., description="Sweep range maximum")
    n_points: int = Field(10, ge=2, le=200, description="Number of sweep points")


class OperatingPointInput(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)


class MultiRunRequest(BaseModel):
    """Multiple arbitrary sizing jobs."""

    points: List[OperatingPointInput] = Field(
        ..., min_length=1, max_length=500, description="List of operating points"
    )


class SensitivityRequest(BaseModel):
    """Sensitivity matrix (Jacobian) input."""

    flow_rate: float = Field(..., gt=0, description="Base Q [m³/s]")
    head: float = Field(..., gt=0, description="Base H [m]")
    rpm: float = Field(..., gt=0, description="Base RPM")
    variables: List[str] = Field(
        default=["flow_rate", "head", "rpm"],
        description="Input variables to perturb",
    )
    perturbation_pct: float = Field(
        1.0, gt=0, le=50, description="Perturbation percentage for finite differences"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sizing_to_dict(s: Any) -> Dict[str, Any]:
    """Convert a SizingResult to a JSON-friendly dict.

    Args:
        s: SizingResult dataclass.

    Returns:
        Dictionary with key sizing outputs.
    """
    return {
        "specific_speed_nq": s.specific_speed_nq,
        "impeller_d2": s.impeller_d2,
        "impeller_d1": s.impeller_d1,
        "impeller_b2": s.impeller_b2,
        "blade_count": s.blade_count,
        "beta1": s.beta1,
        "beta2": s.beta2,
        "estimated_efficiency": s.estimated_efficiency,
        "estimated_power": s.estimated_power,
        "estimated_npsh_r": s.estimated_npsh_r,
        "sigma": s.sigma,
        "warnings": s.warnings,
    }


def _run_one(flow_rate: float, head: float, rpm: float) -> Dict[str, Any]:
    """Run a single sizing and return dict.

    Args:
        flow_rate: Q [m³/s].
        head: H [m].
        rpm: Rotational speed [rev/min].

    Returns:
        Sizing result as a dictionary.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=flow_rate, head=head, rpm=rpm)
    return _sizing_to_dict(run_sizing(op))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/batch/parametric")
def parametric_sweep(req: ParametricRequest) -> Dict[str, Any]:
    """Run a parametric sweep over one variable.

    Varies *sweep_variable* from sweep_min to sweep_max in n_points steps
    while holding the other operating parameters constant.

    Args:
        req: ParametricRequest with base params and sweep definition.

    Returns:
        Dict with sweep_variable, sweep_values, and list of sizing results.
    """
    allowed = {"flow_rate", "head", "rpm"}
    if req.sweep_variable not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"sweep_variable must be one of {allowed}",
        )

    values = np.linspace(req.sweep_min, req.sweep_max, req.n_points).tolist()
    results: List[Dict[str, Any]] = []

    for val in values:
        params = {
            "flow_rate": req.flow_rate,
            "head": req.head,
            "rpm": req.rpm,
        }
        params[req.sweep_variable] = val
        try:
            results.append(_run_one(**params))
        except Exception as exc:
            results.append({"error": str(exc), **params})

    return {
        "sweep_variable": req.sweep_variable,
        "sweep_values": values,
        "n_points": len(values),
        "results": results,
    }


@router.post("/batch/multi_run")
def multi_run(req: MultiRunRequest) -> Dict[str, Any]:
    """Run multiple arbitrary sizing jobs.

    Each entry in *points* is an independent operating point.

    Args:
        req: MultiRunRequest with a list of operating points.

    Returns:
        Dict with n_points and ordered list of sizing results.
    """
    results: List[Dict[str, Any]] = []
    for pt in req.points:
        try:
            results.append(_run_one(pt.flow_rate, pt.head, pt.rpm))
        except Exception as exc:
            results.append({
                "error": str(exc),
                "flow_rate": pt.flow_rate,
                "head": pt.head,
                "rpm": pt.rpm,
            })

    return {"n_points": len(results), "results": results}


@router.post("/batch/sensitivity_matrix")
def sensitivity_matrix(req: SensitivityRequest) -> Dict[str, Any]:
    """Compute a Jacobian sensitivity matrix via central finite differences.

    Perturbs each input variable by +-perturbation_pct% around the base
    operating point, evaluating key output responses.

    Args:
        req: SensitivityRequest with base params and perturbation spec.

    Returns:
        Dict with input_variables, output_variables, jacobian (2D list),
        and base_result.
    """
    allowed = {"flow_rate", "head", "rpm"}
    for v in req.variables:
        if v not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Variable '{v}' not in {allowed}",
            )

    base_params = {
        "flow_rate": req.flow_rate,
        "head": req.head,
        "rpm": req.rpm,
    }

    try:
        base_result = _run_one(**base_params)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output_keys = [
        "estimated_efficiency",
        "estimated_npsh_r",
        "estimated_power",
        "impeller_d2",
        "impeller_b2",
    ]

    jacobian: List[List[float]] = []

    for var in req.variables:
        delta = base_params[var] * req.perturbation_pct / 100.0
        if delta == 0:
            delta = 1e-6

        # Forward
        params_fwd = dict(base_params)
        params_fwd[var] = base_params[var] + delta
        try:
            res_fwd = _run_one(**params_fwd)
        except Exception:
            res_fwd = base_result

        # Backward
        params_bwd = dict(base_params)
        params_bwd[var] = base_params[var] - delta
        try:
            res_bwd = _run_one(**params_bwd)
        except Exception:
            res_bwd = base_result

        row: List[float] = []
        for okey in output_keys:
            fwd_val = res_fwd.get(okey, 0.0)
            bwd_val = res_bwd.get(okey, 0.0)
            deriv = (fwd_val - bwd_val) / (2.0 * delta) if delta != 0 else 0.0
            row.append(float(deriv))
        jacobian.append(row)

    return {
        "input_variables": req.variables,
        "output_variables": output_keys,
        "jacobian": jacobian,
        "base_result": base_result,
        "perturbation_pct": req.perturbation_pct,
    }
