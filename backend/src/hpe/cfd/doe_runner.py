"""Runner de DoE paramétrico com alimentação do surrogate — Fase 15.

Executa um plano DoE (gerado por :mod:`hpe.cfd.doe`) em paralelo via
Celery ou ThreadPoolExecutor, insere resultados no training_log e
dispara retreino do surrogate quando atingir N novos registros.

Usage
-----
    from hpe.cfd.doe_runner import run_doe_campaign, DoECampaignConfig

    config = DoECampaignConfig(n_samples=16, run_solver=True, retrain_after=8)
    summary = run_doe_campaign(sizing_bep, design_space, config)
    print(summary.n_completed, summary.retrain_triggered)
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpe.core.models import SizingResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class DoECampaignConfig:
    """Parâmetros para campanha DoE.

    Attributes
    ----------
    n_samples : int
        Número de pontos DoE a simular.
    sampling_method : str
        "lhs" | "sobol" | "random" | "factorial".
    run_solver : bool
        Se True, executa OpenFOAM em cada ponto.
    n_procs : int
        Processadores por simulação.
    max_workers : int
        Simulações em paralelo.
    mesh_mode : str
        "snappy" | "structured_blade".
    turbulence_model : str
        "kEpsilon" | "kOmegaSST".
    n_iter : int
        Iterações máximas por simulação.
    retrain_after : int | None
        Disparar retreino do surrogate após este número de novos registros.
        None = não retreinar.
    output_dir : str
        Diretório raiz para casos CFD.
    seed : int | None
        Semente para reprodutibilidade.
    """
    n_samples: int = 16
    sampling_method: str = "lhs"
    run_solver: bool = False
    n_procs: int = 4
    max_workers: int = 1
    mesh_mode: str = "snappy"
    turbulence_model: str = "kEpsilon"
    n_iter: int = 500
    retrain_after: Optional[int] = None
    output_dir: str = "doe_campaign"
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Resultado da campanha
# ---------------------------------------------------------------------------

@dataclass
class DoEPointResult:
    """Resultado de um único ponto DoE."""
    point_id: str
    design_values: dict
    converged: bool = False
    H_cfd: Optional[float] = None
    eta_cfd: Optional[float] = None
    P_shaft: Optional[float] = None
    training_log_id: Optional[str] = None
    case_dir: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "point_id": self.point_id,
            "design": self.design_values,
            "converged": self.converged,
            "H_cfd": self.H_cfd,
            "eta_cfd": round(self.eta_cfd, 4) if self.eta_cfd else None,
            "P_shaft_kW": round(self.P_shaft / 1000, 3) if self.P_shaft else None,
            "training_log_id": self.training_log_id,
            "error": self.error,
        }


@dataclass
class DoECampaignSummary:
    """Resumo da campanha DoE."""
    campaign_id: str
    n_planned: int
    n_completed: int
    n_converged: int
    n_failed: int
    retrain_triggered: bool
    retrain_rmse: Optional[float]
    results: list[DoEPointResult]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "n_planned": self.n_planned,
            "n_completed": self.n_completed,
            "n_converged": self.n_converged,
            "n_failed": self.n_failed,
            "retrain_triggered": self.retrain_triggered,
            "retrain_rmse": self.retrain_rmse,
            "errors": self.errors,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def run_doe_campaign(
    sizing_bep,
    design_space,
    config: DoECampaignConfig,
) -> DoECampaignSummary:
    """Executar campanha DoE completa.

    Parameters
    ----------
    sizing_bep : SizingResult
        Ponto de projeto nominal (BEP).
    design_space : DesignSpace
        Espaço de busca com variáveis e limites.
    config : DoECampaignConfig
        Parâmetros da campanha.

    Returns
    -------
    DoECampaignSummary
        Resumo com todos os resultados.
    """
    from hpe.cfd.doe import generate_lhs, generate_sobol, generate_random, generate_full_factorial

    campaign_id = uuid.uuid4().hex[:10]
    output_dir = Path(config.output_dir) / campaign_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "DoE campaign %s: %d points, method=%s, solver=%s",
        campaign_id, config.n_samples, config.sampling_method, config.run_solver,
    )

    # Gerar plano DoE
    generators = {
        "lhs": generate_lhs,
        "sobol": generate_sobol,
        "random": generate_random,
        "factorial": lambda s, n, seed=None: generate_full_factorial(s, levels=max(2, int(n ** (1 / s.ndim)))),
    }
    gen_fn = generators.get(config.sampling_method, generate_lhs)
    doe_points = gen_fn(design_space, config.n_samples, seed=config.seed)

    # Executar pontos
    results: list[DoEPointResult] = []
    if config.max_workers > 1:
        results = _run_parallel(doe_points, sizing_bep, config, output_dir)
    else:
        for dp in doe_points:
            r = _run_doe_point(dp, sizing_bep, config, output_dir)
            results.append(r)

    n_conv = sum(1 for r in results if r.converged)
    n_fail = sum(1 for r in results if r.error)
    errors = [r.error for r in results if r.error]

    # Retreino do surrogate
    retrain_triggered = False
    retrain_rmse = None
    if config.retrain_after is not None and n_conv >= config.retrain_after:
        try:
            retrain_rmse = _trigger_surrogate_retrain()
            retrain_triggered = True
            log.info("DoE: surrogate retrain triggered after %d converged points", n_conv)
        except Exception as exc:
            log.warning("DoE: surrogate retrain failed: %s", exc)
            errors.append(f"Surrogate retrain failed: {exc}")

    summary = DoECampaignSummary(
        campaign_id=campaign_id,
        n_planned=len(doe_points),
        n_completed=len(results),
        n_converged=n_conv,
        n_failed=n_fail,
        retrain_triggered=retrain_triggered,
        retrain_rmse=retrain_rmse,
        results=results,
        errors=errors,
    )
    log.info(
        "DoE campaign %s done: %d/%d converged, retrain=%s",
        campaign_id, n_conv, len(doe_points), retrain_triggered,
    )
    return summary


# ---------------------------------------------------------------------------
# Execução por ponto
# ---------------------------------------------------------------------------

def _run_doe_point(
    design_point,
    sizing_bep,
    config: DoECampaignConfig,
    output_dir: Path,
) -> DoEPointResult:
    """Rodar CFD para um ponto DoE."""
    point_id = uuid.uuid4().hex[:8]
    case_dir = output_dir / point_id

    result = DoEPointResult(
        point_id=point_id,
        design_values=design_point.to_dict(),
        case_dir=str(case_dir),
    )

    try:
        # Construir sizing modificado com os valores do ponto DoE
        modified_sizing = _apply_design_point(sizing_bep, design_point)

        from hpe.cfd.pipeline import run_cfd_pipeline
        cfd_result = run_cfd_pipeline(
            sizing=modified_sizing,
            output_dir=case_dir,
            run_solver=config.run_solver,
            n_procs=config.n_procs,
            mesh_mode=config.mesh_mode,
            turbulence_model=config.turbulence_model,
            n_iter=config.n_iter,
        )

        if cfd_result.performance:
            result.H_cfd = cfd_result.performance.head
            result.eta_cfd = cfd_result.performance.eta_total
            result.P_shaft = cfd_result.performance.shaft_power
            result.converged = cfd_result.ran_simulation
        result.training_log_id = cfd_result.training_log_id

    except Exception as exc:
        result.error = str(exc)
        log.error("DoE point %s failed: %s", point_id, exc)

    return result


def _run_parallel(
    doe_points: list,
    sizing_bep,
    config: DoECampaignConfig,
    output_dir: Path,
) -> list[DoEPointResult]:
    """Executar pontos em paralelo via ThreadPoolExecutor."""
    results = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(_run_doe_point, dp, sizing_bep, config, output_dir): dp
            for dp in doe_points
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                dp = futures[future]
                results.append(DoEPointResult(
                    point_id="error",
                    design_values=dp.to_dict(),
                    error=str(exc),
                ))
    return results


def _apply_design_point(sizing_bep, design_point) -> "SizingResult":  # type: ignore[name-defined]
    """Criar SizingResult modificado com os parâmetros do ponto DoE.

    Cria uma cópia rasa do sizing_bep e substitui os atributos
    correspondentes às variáveis de projeto.
    """
    import copy
    modified = copy.copy(sizing_bep)

    for name, value in design_point.values.items():
        if hasattr(modified, name):
            object.__setattr__(modified, name, value)
        # Também propagar para params se existir
        if hasattr(modified, "params") and hasattr(modified.params, name):
            object.__setattr__(modified.params, name, value)

    return modified


def _trigger_surrogate_retrain() -> Optional[float]:
    """Disparar retreino do surrogate v1 (XGBoost) com dados recentes.

    Delega à camada central ``hpe.data.ingest.retrain_surrogate``, que
    reconstrói a matriz de features combinada (bancada + training_log) e
    chama ``SurrogateV1.train`` com o ``features_path`` correto.

    NOTA: a versão anterior chamava ``SurrogateV1().train()`` sem o argumento
    obrigatório ``features_path`` — o retreino sempre falhava.
    """
    from hpe.data.ingest import retrain_surrogate
    report = retrain_surrogate()
    if report is None:
        raise RuntimeError("retrain_surrogate retornou None (sem dados ou deps ausentes)")
    return report.get("rmse_pct")
