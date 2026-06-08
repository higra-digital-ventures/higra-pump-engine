"""API routes for CFD design loop.

Provides endpoints to set up, run, and retrieve results from
automated CFD design iterations.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cfd", tags=["cfd"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CFDSetupRequest(BaseModel):
    """Request body for POST /cfd/setup."""

    flow_rate: float = Field(..., gt=0, description="Flow rate [m3/s]")
    head: float = Field(..., gt=0, description="Design head [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")
    target_head: Optional[float] = Field(None, description="Head target for comparison [m]")
    target_efficiency: Optional[float] = Field(None, description="Efficiency target [-]")
    n_procs: int = Field(4, ge=1, description="Number of parallel processors")
    timeout: Optional[int] = Field(None, description="Solver timeout [s]")
    machine_type: str = Field("centrifugal_pump", description="Machine type")


class CFDRunRequest(BaseModel):
    """Request body for POST /cfd/run."""

    run_id: str = Field(..., description="Run ID from /cfd/setup")


class CFDDesignLoopRequest(BaseModel):
    """Request body for POST /cfd/design_loop."""

    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    target_head: Optional[float] = None
    target_efficiency: Optional[float] = None
    max_iterations: int = Field(5, ge=1, le=20)
    head_tolerance: float = Field(0.02, gt=0)
    eta_tolerance: float = Field(0.01, gt=0)
    n_procs: int = Field(4, ge=1)
    timeout: Optional[int] = None


class CFDResultsResponse(BaseModel):
    """Response schema for CFD results."""

    head: float
    efficiency: float
    power: float
    pressure_rise: float
    total_pressure_loss: float
    mass_flow_check: float
    blade_loading: list[dict[str, Any]]
    convergence_residuals: list[float]


class DesignLoopResponse(BaseModel):
    """Response schema for the full design loop."""

    run_id: str
    converged: bool
    n_iterations: int
    final_head: float
    final_efficiency: float
    final_power: float
    history: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# In-memory store for runs (production would use DB)
# ---------------------------------------------------------------------------

_runs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/setup")
def cfd_setup(req: CFDSetupRequest) -> dict[str, Any]:
    """Prepare a CFD case from sizing parameters.

    Runs meanline sizing, builds the OpenFOAM case directory, and
    returns a ``run_id`` for subsequent operations.
    """
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing
    from hpe.cfd.design_loop import CFDDesignLoop

    op = OperatingPoint(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        machine_type=MachineType(req.machine_type),
    )
    sizing = run_sizing(op)

    run_id = uuid.uuid4().hex[:12]
    work_dir = Path("cfd_runs") / run_id

    loop = CFDDesignLoop(work_dir=work_dir)
    loop.setup(
        sizing_result=sizing,
        solver_params={"n_procs": req.n_procs, "timeout": req.timeout},
        target_head=req.target_head,
        target_efficiency=req.target_efficiency,
    )

    _runs[run_id] = {
        "loop": loop,
        "sizing": sizing,
        "status": "ready",
        "results": None,
        "work_dir": work_dir,
    }

    return {
        "run_id": run_id,
        "status": "ready",
        "case_dir": str(work_dir),
        "target_head": loop._target_head,
        "target_efficiency": loop._target_efficiency,
    }


@router.post("/run")
def cfd_run(req: CFDRunRequest) -> dict[str, Any]:
    """Execute a single CFD run for a previously set-up case.

    Returns extracted results.
    """
    if req.run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {req.run_id} not found.")

    entry = _runs[req.run_id]
    loop: Any = entry["loop"]
    sizing = entry["sizing"]

    case_dir = loop.work_dir / "single_run"

    from hpe.cfd.openfoam.case_builder import build_case
    build_case(
        sizing=sizing,
        step_file=case_dir / "placeholder.step",
        output_dir=case_dir,
        n_procs=loop._solver_params.get("n_procs", 4),
    )

    try:
        run_results = loop.run_single(case_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not all(r.success for r in run_results):
        failed = [r for r in run_results if not r.success]
        raise HTTPException(
            status_code=500,
            detail={
                "message": "CFD run failed",
                "failed_steps": [
                    {"command": r.command, "stderr": r.stderr[:500]}
                    for r in failed
                ],
            },
        )

    cfd_results = loop.extract_results(case_dir)
    entry["results"] = cfd_results
    entry["status"] = "completed"

    return {
        "run_id": req.run_id,
        "status": "completed",
        "results": {
            "head": round(cfd_results.head, 4),
            "efficiency": round(cfd_results.efficiency, 4),
            "power": round(cfd_results.power, 2),
            "pressure_rise": round(cfd_results.pressure_rise, 2),
            "total_pressure_loss": round(cfd_results.total_pressure_loss, 2),
            "mass_flow_check": cfd_results.mass_flow_check,
            "n_blade_loading_spans": len(cfd_results.blade_loading),
            "n_residual_points": len(cfd_results.convergence_residuals),
        },
    }


@router.post("/design_loop")
def cfd_design_loop(req: CFDDesignLoopRequest) -> DesignLoopResponse:
    """Run the full automated CFD design loop.

    Iterates sizing -> CFD -> compare -> adjust until convergence.
    """
    from hpe.core.enums import MachineType
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing
    from hpe.cfd.design_loop import CFDDesignLoop

    op = OperatingPoint(
        flow_rate=req.flow_rate,
        head=req.head,
        rpm=req.rpm,
        machine_type=MachineType("centrifugal_pump"),
    )
    sizing = run_sizing(op)

    run_id = uuid.uuid4().hex[:12]
    work_dir = Path("cfd_runs") / run_id

    loop = CFDDesignLoop(work_dir=work_dir)
    loop.setup(
        sizing_result=sizing,
        solver_params={"n_procs": req.n_procs, "timeout": req.timeout},
        target_head=req.target_head,
        target_efficiency=req.target_efficiency,
    )

    try:
        result = loop.run_design_loop(
            max_iterations=req.max_iterations,
            head_tolerance=req.head_tolerance,
            eta_tolerance=req.eta_tolerance,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    history_out = [
        {
            "iteration": h.iteration,
            "head_target": round(h.head_target, 4),
            "head_cfd": round(h.head_cfd, 4),
            "eta_target": round(h.eta_target, 4),
            "eta_cfd": round(h.eta_cfd, 4),
            "geometry_changes": h.geometry_changes,
        }
        for h in result.history
    ]

    return DesignLoopResponse(
        run_id=result.run_id,
        converged=result.converged,
        n_iterations=result.n_iterations,
        final_head=round(result.final_results.head, 4),
        final_efficiency=round(result.final_results.efficiency, 4),
        final_power=round(result.final_results.power, 2),
        history=history_out,
    )


@router.get("/results/{run_id}")
def get_cfd_results(run_id: str) -> dict[str, Any]:
    """Retrieve results from a previously executed CFD run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    entry = _runs[run_id]
    if entry["results"] is None:
        return {"run_id": run_id, "status": entry["status"], "results": None}

    r = entry["results"]
    return {
        "run_id": run_id,
        "status": entry["status"],
        "results": {
            "head": round(r.head, 4),
            "efficiency": round(r.efficiency, 4),
            "power": round(r.power, 2),
            "pressure_rise": round(r.pressure_rise, 2),
            "total_pressure_loss": round(r.total_pressure_loss, 2),
            "mass_flow_check": r.mass_flow_check,
            "blade_loading": r.blade_loading,
            "convergence_residuals": r.convergence_residuals,
        },
    }


# ===========================================================================
# Fase 11 — Multi-point sweep + pump curve
# ===========================================================================

class SweepRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Q_bep [m³/s]")
    head: float = Field(..., gt=0, description="H_bep [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")
    flow_fractions: list[float] = Field(
        default=[0.50, 0.70, 0.85, 1.00, 1.15, 1.30],
        description="Frações de Q_bep a simular",
    )
    run_solver: bool = Field(False, description="Executar solver OpenFOAM")
    n_procs: int = Field(4, ge=1)
    turbulence_model: str = Field("kEpsilon", description="kEpsilon | kOmegaSST")
    mesh_mode: str = Field("snappy", description="snappy | structured_blade")
    n_iter: int = Field(500, ge=100)


@router.post("/sweep", summary="Multi-point operating sweep (pump curve)")
def cfd_sweep(req: SweepRequest) -> dict[str, Any]:
    """Executar varredura CFD multi-ponto e retornar curva H-Q + η-Q.

    Simula ``flow_fractions`` × Q_bep em paralelo (ou série) e ajusta
    polinômio grau 2 nos resultados para construir a curva da bomba.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.cfd.sweep import run_cfd_sweep, SweepConfig
    from hpe.cfd.pump_curve import build_pump_curve

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    config = SweepConfig(
        flow_fractions=req.flow_fractions,
        run_solver=req.run_solver,
        n_procs=req.n_procs,
        mesh_mode=req.mesh_mode,
        turbulence_model=req.turbulence_model,
        n_iter=req.n_iter,
    )

    import tempfile
    work_dir = Path(tempfile.gettempdir()) / f"hpe_sweep_{uuid.uuid4().hex[:8]}"
    sweep = run_cfd_sweep(sizing, config, work_dir)

    curve = build_pump_curve(sweep)
    return {
        "sweep": sweep.to_dict(),
        "pump_curve": curve.to_dict(),
    }


@router.post("/pump_curve", summary="Build pump curve from existing sweep results")
def build_curve_from_points(body: dict[str, Any]) -> dict[str, Any]:
    """Construir curva da bomba a partir de listas de pontos Q/H/η/P.

    Aceita dados de bancada, CFD ou estimados.

    Body JSON::

        {
          "Q": [0.03, 0.04, 0.05, 0.06],
          "H": [35, 32, 30, 26],
          "eta": [0.72, 0.78, 0.80, 0.76],
          "P_kW": [14, 16, 18, 19],
          "n_rpm": 1750
        }
    """
    from hpe.cfd.pump_curve import build_pump_curve_from_points

    Q = body.get("Q", [])
    H = body.get("H", [])
    eta = body.get("eta", [])
    P = [p * 1000 for p in body.get("P_kW", [])]
    n_rpm = float(body.get("n_rpm", 1750))

    if len(Q) < 3:
        raise HTTPException(status_code=422, detail="Mínimo 3 pontos necessários")

    # Se P não fornecido, estimar
    if not P:
        from hpe.cfd.pump_curve import _estimate_power
        P = [_estimate_power(q, h, e) for q, h, e in zip(Q, H, eta or [0.8] * len(Q))]

    if not eta:
        eta = [0.80] * len(Q)

    curve = build_pump_curve_from_points(
        Q_pts=Q, H_pts=H, eta_pts=eta, P_pts=P, n_rpm=n_rpm
    )
    return curve.to_dict()


# ===========================================================================
# Fase 13 — Blade loading + Cavitation
# ===========================================================================

class CavitationRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    npsh_available: float = Field(..., gt=0, description="NPSHa [m]")
    fluid_temp_c: float = Field(20.0, description="Temperatura fluido [°C]")
    safety_margin: float = Field(0.5, description="Margem mínima NPSHa-NPSHr [m]")
    flow_fractions: list[float] = Field(
        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        description="Frações de Q_bep para curva NPSHr-Q",
    )


@router.post("/cavitation", summary="Cavitation assessment (NPSHr, Thoma sigma) + NPSHr-Q curve")
def assess_cavitation(req: CavitationRequest) -> dict[str, Any]:
    """Avaliar risco de cavitação e retornar curva NPSHr-Q multi-ponto.

    Além do ponto BEP, calcula NPSHr para cada fração de Q usando a
    correlação de Gülich (§6.10) com head estimado via parábola.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.cfd.results.cavitation import assess_cavitation as _assess

    op_bep = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing_bep = run_sizing(op_bep)

    result_bep = _assess(
        sizing_bep,
        npsh_available=req.npsh_available,
        fluid_temp_c=req.fluid_temp_c,
        safety_margin=req.safety_margin,
    )
    out = result_bep.to_dict()

    # NPSHr-Q curve: re-size at each Q fraction and compute NPSHr
    npshq_curve = []
    for f in req.flow_fractions:
        try:
            q_f = req.flow_rate * f
            # Head estimate via Gülich parabola: H ≈ H_bep*(1.25 - 0.05*f - 0.20*f²)
            h_f = req.head * max(0.3, 1.25 - 0.05 * f - 0.20 * f * f)
            op_f = OperatingPoint(flow_rate=q_f, head=h_f, rpm=req.rpm)
            s_f = run_sizing(op_f)
            r_f = _assess(s_f, npsh_available=req.npsh_available,
                          fluid_temp_c=req.fluid_temp_c, safety_margin=req.safety_margin)
            npshq_curve.append({
                "Q": round(q_f * 3600, 3),    # m³/h for display
                "Q_m3s": round(q_f, 6),
                "fraction": round(f, 2),
                "npsh_r": round(r_f.npsh_r, 3),
                "safe": r_f.safe,
            })
        except Exception:
            pass  # skip points that fail to size

    out["npshq_curve"] = npshq_curve
    return out


class BladeLoadingRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    case_dir: Optional[str] = Field(None, description="Diretório caso OpenFOAM (opcional)")
    n_chord: int = Field(21, ge=5, le=100)


@router.post("/blade_loading", summary="Blade loading diagram (Cp distribution)")
def blade_loading(req: BladeLoadingRequest) -> dict[str, Any]:
    """Calcular diagrama de carregamento da pá (Cp PS e SS ao longo da corda).

    Se ``case_dir`` aponta para um caso OpenFOAM com resultados CFD,
    extrai Cp dos dados CFD.  Caso contrário, usa estimativa analítica.
    """
    from hpe.core.models import OperatingPoint
    from hpe.cfd.results.blade_loading import extract_blade_loading

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    case = Path(req.case_dir) if req.case_dir else Path(".")

    result = extract_blade_loading(case, op, n_chord=req.n_chord)
    return result.to_dict()


# ===========================================================================
# Fase 14 — SU2 Adjoint
# ===========================================================================

class SU2AdjointRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    case_dir: str = Field(..., description="Diretório do caso SU2 (com config.cfg e mesh)")
    n_procs: int = Field(1, ge=1)
    run_direct_first: bool = Field(True, description="Rodar análise direta antes do adjoint")
    objective: str = Field("total_pressure_loss", description="Função objetivo SU2")


@router.post("/su2/adjoint", summary="SU2 adjoint — compute shape sensitivities")
def su2_adjoint(req: SU2AdjointRequest) -> dict[str, Any]:
    """Executar SU2 adjoint e retornar sensibilidades normalizadas das variáveis de projeto.

    Requer SU2_CFD e SU2_CFD_AD no PATH.  Retorna erro 503 se não disponível.
    """
    from hpe.cfd.su2.runner import su2_available, run_su2_direct, run_su2_adjoint
    from hpe.cfd.su2.sensitivity import extract_from_runner_result
    from hpe.sizing.meanline import run_sizing
    from hpe.core.models import OperatingPoint

    if not su2_available():
        raise HTTPException(
            status_code=503,
            detail="SU2_CFD not found in PATH. Install SU2 and add to PATH.",
        )

    case_dir = Path(req.case_dir)
    config_path = case_dir / "config.cfg"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"config.cfg not found in {case_dir}")

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    direct_result = None
    if req.run_direct_first:
        direct_result = run_su2_direct(config_path, work_dir=case_dir, n_procs=req.n_procs)
        if not direct_result.converged:
            return {
                "status": "direct_not_converged",
                "direct": direct_result.to_dict(),
                "adjoint": None,
                "sensitivities": None,
            }

    adjoint_result = run_su2_adjoint(
        config_path,
        direct_solution=direct_result.solution_file if direct_result else None,
        work_dir=case_dir,
        n_procs=req.n_procs,
    )

    sens = extract_from_runner_result(adjoint_result, sizing)
    sens.objective = req.objective

    return {
        "status": "ok" if adjoint_result.converged else "not_converged",
        "direct": direct_result.to_dict() if direct_result else None,
        "adjoint": adjoint_result.to_dict(),
        "sensitivities": sens.to_dict(),
        "steepest_descent": sens.steepest_descent_step(step_size=0.01),
    }


# ===========================================================================
# Fase 15 — DoE parametric campaign
# ===========================================================================

class DoERequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    n_samples: int = Field(16, ge=4, le=200)
    sampling_method: str = Field("lhs", description="lhs | sobol | random | factorial")
    variables: list[str] = Field(
        default=["beta1", "beta2", "d2", "b2"],
        description="Variáveis de projeto a incluir no DoE",
    )
    variation: float = Field(0.15, gt=0, lt=0.5, description="Variação relativa ±")
    run_solver: bool = Field(False)
    n_procs: int = Field(4, ge=1)
    max_workers: int = Field(1, ge=1)
    turbulence_model: str = Field("kEpsilon")
    retrain_after: Optional[int] = Field(None, description="Retreinar surrogate após N pontos CFD")
    seed: Optional[int] = Field(42)


@router.post("/doe", summary="Parametric DoE campaign (LHS / Sobol / factorial)")
def doe_campaign(req: DoERequest) -> dict[str, Any]:
    """Executar campanha DoE multi-variável sobre o espaço de projeto da bomba.

    Gera plano amostral (LHS por padrão), executa pipeline CFD em cada ponto
    e insere resultados no training_log.  Opcionalmente retreina o surrogate.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.cfd.doe import DesignSpace
    from hpe.cfd.doe_runner import DoECampaignConfig, run_doe_campaign

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    space = DesignSpace.from_sizing(
        sizing,
        variation=req.variation,
        include=req.variables,
    )

    config = DoECampaignConfig(
        n_samples=req.n_samples,
        sampling_method=req.sampling_method,
        run_solver=req.run_solver,
        n_procs=req.n_procs,
        max_workers=req.max_workers,
        turbulence_model=req.turbulence_model,
        retrain_after=req.retrain_after,
        seed=req.seed,
    )

    summary = run_doe_campaign(sizing, space, config)
    return summary.to_dict()


# ===========================================================================
# Cancelamento de job CFD
# ===========================================================================

@router.delete("/run/{run_id}", summary="Cancel / stop a CFD run")
def cancel_cfd_run(run_id: str) -> dict[str, Any]:
    """Solicitar parada de um run CFD ativo.

    Escreve o arquivo ``system/stopAt`` no case dir para sinalizar ao
    OpenFOAM que deve encerrar na próxima iteração.  Se o run não existe
    ou já terminou, retorna status atual sem erro.
    """
    entry = _runs.get(run_id)
    if entry is None:
        return {"run_id": run_id, "status": "not_found", "cancelled": False}

    current_status = entry.get("status", "unknown")
    if current_status in ("completed", "cancelled", "failed"):
        return {"run_id": run_id, "status": current_status, "cancelled": False}

    # Escrever arquivo de parada para o OpenFOAM
    work_dir: Path = entry.get("work_dir", Path("cfd_runs") / run_id)
    try:
        from hpe.cfd.openfoam.convergence import ConvergenceMonitor, ConvergenceCriteria
        monitor = ConvergenceMonitor(case_dir=work_dir, criteria=ConvergenceCriteria())
        monitor.write_stop_file()
    except Exception as exc:
        log.warning("cancel_cfd_run %s: could not write stop file — %s", run_id, exc)

    entry["status"] = "cancelled"
    log.info("CFD run %s cancelled by API request", run_id)
    return {"run_id": run_id, "status": "cancelled", "cancelled": True}


# ===========================================================================
# Fase 16 — Loop adjoint fechado
# ===========================================================================

class AdjointLoopRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Q_bep [m³/s]")
    head: float = Field(..., gt=0, description="H_bep [m]")
    rpm: float = Field(..., gt=0, description="Rotação [rpm]")
    max_iter: int = Field(5, ge=1, le=20, description="Iterações máximas do loop")
    step_size: float = Field(0.02, gt=0, lt=1.0, description="Passo de descida normalizado")
    tol: float = Field(1e-3, gt=0, description="Tolerância de convergência |∇J|")
    n_procs: int = Field(1, ge=1)
    turbulence_model: str = Field("kEpsilon")
    mesh_mode: str = Field("snappy")
    design_vars: list[str] = Field(
        default=["beta2", "d2", "b2"],
        description="Variáveis de projeto a otimizar",
    )
    objective: str = Field("total_pressure_loss", description="Função objetivo SU2")


@router.post("/adjoint/loop", summary="Closed adjoint optimization loop (Fase 16)")
def run_adjoint_loop(req: AdjointLoopRequest) -> dict[str, Any]:
    """Executar loop de otimização adjoint fechado.

    Ciclo: SU2 direto → adjoint → sensibilidades → descida → modificar sizing
    → regenerar geometria/malha → repetir até convergência ou max_iter.

    Quando SU2 não está instalado, executa em modo sintético (útil para
    testes e desenvolvimento).
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.cfd.adjoint_loop import AdjointConfig, run_adjoint_loop as _run

    import tempfile

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    config = AdjointConfig(
        max_iter=req.max_iter,
        step_size=req.step_size,
        tol=req.tol,
        n_procs=req.n_procs,
        turbulence_model=req.turbulence_model,
        mesh_mode=req.mesh_mode,
        design_vars=req.design_vars,
        objective=req.objective,
        output_dir=str(Path(tempfile.gettempdir()) / "hpe_adjoint"),
    )

    result = _run(sizing, case_dir=Path(tempfile.gettempdir()) / "hpe_adjoint", config=config)
    return result.to_dict()
