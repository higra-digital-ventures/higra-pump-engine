"""API routes for Reactive Response Surface (RRS) optimization.

Provides endpoints to run RRS optimization on pump design variables
with configurable objectives and constraints.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/optimize", tags=["optimization"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class VariableBound(BaseModel):
    """Bound for a single design variable."""

    name: str = Field(..., description="Variable name (e.g. 'd2', 'beta2')")
    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")


class RRSRequest(BaseModel):
    """Request body for POST /optimize/rrs."""

    flow_rate: float = Field(..., gt=0, description="Design flow rate [m3/s]")
    head: float = Field(..., gt=0, description="Design head [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")
    objectives: list[str] = Field(
        default=["efficiency"],
        description="Objective names: efficiency, npsh, power",
    )
    variables: list[VariableBound] = Field(
        ...,
        description="Design variables with bounds",
    )
    max_evals: int = Field(50, ge=5, le=500, description="Evaluation budget")
    convergence_tol: float = Field(1e-3, gt=0, description="Convergence tolerance")
    n_initial: Optional[int] = Field(None, description="Initial DoE size")
    seed: int = Field(42, description="Random seed")


class RRSResponse(BaseModel):
    """Response schema for RRS optimization."""

    best_point: dict[str, float]
    best_value: float
    n_evaluations: int
    converged: bool
    surrogate_r2: float
    convergence_history: list[float]
    all_evaluations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/rrs", response_model=RRSResponse)
def run_rrs_optimization(req: RRSRequest) -> RRSResponse:
    """Run Reactive Response Surface optimization on pump design.

    Uses adaptive surrogate modelling (RBF + Expected Improvement)
    to efficiently explore the design space.
    """
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint
    from hpe.optimization.rrs import ReactiveResponseSurface
    from hpe.sizing import run_sizing

    variable_names = [v.name for v in req.variables]
    bounds = [(v.lower, v.upper) for v in req.variables]

    # Build the objective function
    primary_objective = req.objectives[0] if req.objectives else "efficiency"

    def objective_fn(x: list[float]) -> float:
        """Evaluate pump design at the given variable values.

        Maps variable names to sizing overrides, runs sizing, and
        returns the objective value.
        """
        overrides: dict[str, float] = dict(zip(variable_names, x))

        op = OperatingPoint(
            flow_rate=req.flow_rate,
            head=req.head,
            rpm=req.rpm,
            machine_type=MachineType("centrifugal_pump"),
            override_d2=overrides.get("d2"),
            override_b2=overrides.get("b2"),
            override_d1=overrides.get("d1"),
        )

        try:
            result = run_sizing(op)
        except Exception:
            # Penalize infeasible points
            return 1e6 if primary_objective != "efficiency" else 0.0

        if primary_objective == "efficiency":
            # Maximize efficiency → minimize negative efficiency
            return -result.estimated_efficiency
        elif primary_objective == "npsh":
            # Minimize NPSH required
            return result.estimated_npsh_r
        elif primary_objective == "power":
            # Minimize power
            return result.estimated_power
        else:
            return -result.estimated_efficiency

    # Run optimizer
    optimizer = ReactiveResponseSurface(seed=req.seed)
    rrs_result = optimizer.optimize(
        objective_fn=objective_fn,
        bounds=bounds,
        n_initial=req.n_initial,
        max_evals=req.max_evals,
        convergence_tol=req.convergence_tol,
        minimize_objective=True,
    )

    # Format best point as named dict
    best_named = dict(zip(variable_names, rrs_result.best_point))

    # If maximizing efficiency, negate the stored value back
    best_value = rrs_result.best_value
    conv_history = rrs_result.convergence_history
    all_evals = rrs_result.all_evaluations

    if primary_objective == "efficiency":
        best_value = -best_value
        conv_history = [-v for v in conv_history]
        for ev in all_evals:
            ev["value"] = -ev["value"]

    return RRSResponse(
        best_point={k: round(v, 6) for k, v in best_named.items()},
        best_value=round(best_value, 6),
        n_evaluations=rrs_result.n_evaluations,
        converged=rrs_result.converged,
        surrogate_r2=rrs_result.surrogate_r2,
        convergence_history=[round(v, 6) for v in conv_history],
        all_evaluations=all_evals,
    )
