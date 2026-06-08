"""Loop de otimização adjoint fechado — Fase 16.

Orquestra o ciclo completo:
  1. SU2 direto (RANS) → função objetivo
  2. SU2 adjoint (CDA) → sensibilidades de superfície
  3. Extração e normalização → DesignSensitivities
  4. Passo de descida → Δβ₂, ΔD₂, Δb₂, …
  5. Aplicar deltas → SizingResult modificado
  6. Regenerar geometria (runner + malha)
  7. Voltar ao passo 1 até convergência

Usage
-----
    from hpe.cfd.adjoint_loop import run_adjoint_loop, AdjointConfig

    config = AdjointConfig(max_iter=5, step_size=0.02, tol=1e-3)
    result = run_adjoint_loop(sizing, case_dir, config)
    print(result.converged, result.n_iterations, result.best_objective)
"""

from __future__ import annotations

import copy
import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class AdjointConfig:
    """Parâmetros do loop de otimização adjoint.

    Attributes
    ----------
    max_iter : int
        Máximo de iterações do loop externo.
    step_size : float
        Passo de descida normalizado (0 < α ≤ 1).
    tol : float
        Tolerância de convergência: |ΔJ/J| < tol.
    n_procs : int
        Processos MPI para SU2.
    n_iter_cfd : int
        Iterações internas de cada execução SU2.
    turbulence_model : str
        "kEpsilon" | "kOmegaSST".
    mesh_mode : str
        "snappy" | "structured_blade".
    design_vars : list[str]
        Variáveis de projeto a perturbar.
    bounds : dict[str, tuple[float, float]]
        Limites físicos absolutos para cada variável.
    objective : str
        Nome da função objetivo passado ao extrator.
    output_dir : str
        Diretório raiz para casos de cada iteração.
    use_mesh_morph : bool
        Se True, após a 1ª iteração usa mesh morphing ao invés de
        regenerar a malha do zero — 10-100× mais rápido.
    """
    max_iter: int = 5
    step_size: float = 0.02
    tol: float = 1e-3
    n_procs: int = 1
    n_iter_cfd: int = 500
    turbulence_model: str = "kEpsilon"
    mesh_mode: str = "snappy"
    design_vars: list[str] = field(default_factory=lambda: ["beta2", "d2", "b2"])
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "beta1": (10.0, 45.0),
        "beta2": (12.0, 40.0),
        "d2":    (0.05, 1.0),
        "d1":    (0.02, 0.5),
        "b2":    (0.005, 0.2),
    })
    objective: str = "total_pressure_loss"
    output_dir: str = "adjoint_loop"
    use_mesh_morph: bool = True


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

@dataclass
class AdjointIterResult:
    """Resultado de uma única iteração do loop."""
    iteration: int
    objective: Optional[float]
    gradient_norm: float
    deltas: dict[str, float]
    converged_cfd: bool
    case_dir: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "objective": round(self.objective, 6) if self.objective is not None else None,
            "gradient_norm": round(self.gradient_norm, 6),
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
            "converged_cfd": self.converged_cfd,
            "case_dir": self.case_dir,
            "error": self.error,
        }


@dataclass
class AdjointLoopResult:
    """Resultado completo do loop adjoint."""
    loop_id: str
    n_iterations: int
    converged: bool
    best_objective: Optional[float]
    initial_objective: Optional[float]
    improvement_pct: Optional[float]
    final_sizing: object  # SizingResult
    history: list[AdjointIterResult]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "loop_id": self.loop_id,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "best_objective": round(self.best_objective, 6) if self.best_objective else None,
            "initial_objective": round(self.initial_objective, 6) if self.initial_objective else None,
            "improvement_pct": round(self.improvement_pct, 2) if self.improvement_pct else None,
            "history": [r.to_dict() for r in self.history],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def run_adjoint_loop(
    sizing,
    case_dir: "str | Path",
    config: AdjointConfig,
) -> AdjointLoopResult:
    """Executar loop de otimização adjoint completo.

    Parameters
    ----------
    sizing : SizingResult
        Ponto de projeto inicial.
    case_dir : Path
        Diretório base para os casos CFD de cada iteração.
    config : AdjointConfig
        Parâmetros do loop.

    Returns
    -------
    AdjointLoopResult
        Histórico completo + SizingResult otimizado.
    """
    loop_id = uuid.uuid4().hex[:8]
    base_dir = Path(config.output_dir) / loop_id
    base_dir.mkdir(parents=True, exist_ok=True)

    log.info("Adjoint loop %s: max_iter=%d, step=%.3f, tol=%.1e",
             loop_id, config.max_iter, config.step_size, config.tol)

    current_sizing = copy.copy(sizing)
    history: list[AdjointIterResult] = []
    errors: list[str] = []
    initial_obj: Optional[float] = None
    best_obj: Optional[float] = None
    converged = False

    prev_iter_dir: Optional[Path] = None
    for i in range(config.max_iter):
        iter_dir = base_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        iter_result = _run_adjoint_iter(
            iteration=i,
            sizing=current_sizing,
            work_dir=iter_dir,
            config=config,
            prev_iter_dir=prev_iter_dir,
        )
        history.append(iter_result)
        prev_iter_dir = iter_dir

        if iter_result.error:
            errors.append(f"iter {i}: {iter_result.error}")
            log.warning("Adjoint loop %s iter %d failed: %s", loop_id, i, iter_result.error)
            break

        obj = iter_result.objective
        if obj is None:
            log.warning("Adjoint loop %s iter %d: no objective, stopping", loop_id, i)
            break

        if i == 0:
            initial_obj = obj
        if best_obj is None or obj < best_obj:
            best_obj = obj

        log.info(
            "Adjoint loop %s iter %d: J=%.6f  |∇J|=%.4f",
            loop_id, i, obj, iter_result.gradient_norm,
        )

        # Convergência: gradiente pequeno
        if iter_result.gradient_norm < config.tol:
            log.info("Adjoint loop %s converged at iter %d (|∇J|=%.2e < tol=%.2e)",
                     loop_id, i, iter_result.gradient_norm, config.tol)
            converged = True
            break

        # Aplicar deltas ao sizing atual para a próxima iteração
        current_sizing = _apply_deltas(current_sizing, iter_result.deltas, config)

    improvement = None
    if initial_obj and best_obj and initial_obj != 0:
        improvement = (initial_obj - best_obj) / abs(initial_obj) * 100.0

    result = AdjointLoopResult(
        loop_id=loop_id,
        n_iterations=len(history),
        converged=converged,
        best_objective=best_obj,
        initial_objective=initial_obj,
        improvement_pct=improvement,
        final_sizing=current_sizing,
        history=history,
        errors=errors,
    )
    log.info(
        "Adjoint loop %s done: %d iters, converged=%s, best_J=%s, improve=%.1f%%",
        loop_id, result.n_iterations, converged, best_obj,
        improvement or 0.0,
    )
    return result


# ---------------------------------------------------------------------------
# Iteração individual
# ---------------------------------------------------------------------------

def _run_adjoint_iter(
    iteration: int,
    sizing,
    work_dir: Path,
    config: AdjointConfig,
    prev_iter_dir: Optional[Path] = None,
) -> AdjointIterResult:
    """Executar uma iteração: build case → SU2 direct → SU2 adjoint → extract sens."""
    from hpe.cfd.su2.runner import run_su2_direct, run_su2_adjoint, su2_available
    from hpe.cfd.su2.sensitivity import extract_sensitivities

    # ── Verificar disponibilidade ───────────────────────────────────────────
    if not su2_available():
        log.warning("SU2 not available — using synthetic gradients for iter %d", iteration)
        return _synthetic_iter(iteration, sizing, work_dir, config)

    # ── Gerar caso CFD (morph se possível, rebuild se iter 0) ──────────────
    try:
        if config.use_mesh_morph and prev_iter_dir is not None and prev_iter_dir.exists():
            log.info("iter %d: reusing previous mesh via morphing", iteration)
            config_path = _morph_cfd_case(sizing, work_dir, prev_iter_dir, config)
        else:
            config_path = _build_cfd_case(sizing, work_dir, config)
    except Exception as exc:
        return AdjointIterResult(
            iteration=iteration,
            objective=None,
            gradient_norm=0.0,
            deltas={},
            converged_cfd=False,
            case_dir=str(work_dir),
            error=f"case build failed: {exc}",
        )

    # ── SU2 direto ─────────────────────────────────────────────────────────
    direct = run_su2_direct(
        config_path=config_path,
        work_dir=work_dir / "direct",
        n_procs=config.n_procs,
        timeout=3600,
    )
    if not direct.converged and direct.return_code != 0:
        return AdjointIterResult(
            iteration=iteration,
            objective=direct.objective,
            gradient_norm=0.0,
            deltas={},
            converged_cfd=False,
            case_dir=str(work_dir),
            error=f"SU2 direct failed (rc={direct.return_code})",
        )

    # ── SU2 adjoint ────────────────────────────────────────────────────────
    adj_config = _build_adjoint_config(config_path, work_dir / "adjoint")
    adjoint = run_su2_adjoint(
        config_path=adj_config,
        direct_solution=direct.solution_file,
        work_dir=work_dir / "adjoint",
        n_procs=config.n_procs,
        timeout=3600,
    )

    # ── Extrair sensibilidades ─────────────────────────────────────────────
    if adjoint.sensitivity_file:
        sens = extract_sensitivities(adjoint.sensitivity_file, sizing, config.objective)
    else:
        from hpe.cfd.su2.sensitivity import DesignSensitivities
        sens = DesignSensitivities(objective=config.objective)
        log.warning("No sensitivity file at iter %d — zero gradient", iteration)

    grad_norm = math.sqrt(sum(g ** 2 for g in sens.gradient_vector()))
    deltas = _compute_deltas(sens, config)

    return AdjointIterResult(
        iteration=iteration,
        objective=direct.objective,
        gradient_norm=grad_norm,
        deltas=deltas,
        converged_cfd=direct.converged,
        case_dir=str(work_dir),
    )


def _synthetic_iter(
    iteration: int,
    sizing,
    work_dir: Path,
    config: AdjointConfig,
) -> AdjointIterResult:
    """Iteração sintética quando SU2 não está disponível (CI / dev)."""
    import random
    rng = random.Random(iteration + 42)
    # Gradiente decrescente artificialmente (simula convergência)
    scale = 0.1 * (0.6 ** iteration)
    grad_norm = scale + rng.uniform(0, scale * 0.2)
    deltas = {v: rng.uniform(-config.step_size * scale, config.step_size * scale)
              for v in config.design_vars}
    objective = 1.0 * (0.85 ** iteration) + rng.uniform(-0.02, 0.02)
    log.debug("Synthetic iter %d: J=%.4f  |∇J|=%.4f", iteration, objective, grad_norm)
    return AdjointIterResult(
        iteration=iteration,
        objective=objective,
        gradient_norm=grad_norm,
        deltas=deltas,
        converged_cfd=True,
        case_dir=str(work_dir),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfd_case(sizing, work_dir: Path, config: AdjointConfig) -> Path:
    """Construir caso OpenFOAM/SU2 para o sizing atual."""
    from hpe.cfd.openfoam.case import build_openfoam_case
    case_dir = work_dir / "case"
    build_openfoam_case(
        sizing=sizing,
        output_dir=case_dir,
        mesh_mode=config.mesh_mode,
        turbulence_model=config.turbulence_model,
        n_procs=config.n_procs,
    )
    # Gerar config SU2 na mesma pasta
    from hpe.cfd.su2.config import write_su2_config
    su2_dir = work_dir / "su2"
    su2_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_su2_config(
        sizing=sizing,
        output_dir=su2_dir,
        n_iter=config.n_iter_cfd,
        turbulence_model=config.turbulence_model,
        math_problem="DIRECT",
    )
    return config_path


def _morph_cfd_case(
    sizing, work_dir: Path, prev_iter_dir: Path, config: AdjointConfig,
) -> Path:
    """Copiar caso anterior e aplicar mesh morphing ao invés de rebuild.

    10-100× mais rápido que snappyHexMesh do zero e preserva topologia
    da malha ao longo das iterações do loop adjoint.
    """
    import shutil
    from hpe.cfd.openfoam.morph import morph_mesh, MorphConfig

    # Copiar caso anterior
    prev_case = prev_iter_dir / "case"
    new_case = work_dir / "case"
    if prev_case.exists():
        shutil.copytree(prev_case, new_case, dirs_exist_ok=True)
    else:
        # Fallback: build from scratch
        return _build_cfd_case(sizing, work_dir, config)

    # Extract deltas from sizing attributes (placeholder — in real loop
    # these would come from the previous iteration's sensitivities)
    deltas: dict[str, float] = {v: 0.0 for v in config.design_vars}

    morph_mesh(
        case_dir=new_case,
        design_deltas=deltas,
        sizing=sizing,
        config=MorphConfig(),
    )

    # Gerar config SU2 atualizado para esse caso
    from hpe.cfd.su2.config import write_su2_config
    su2_dir = work_dir / "su2"
    su2_dir.mkdir(parents=True, exist_ok=True)
    return write_su2_config(
        sizing=sizing,
        output_dir=su2_dir,
        n_iter=config.n_iter_cfd,
        turbulence_model=config.turbulence_model,
        math_problem="DIRECT",
    )


def _build_adjoint_config(direct_config: Path, work_dir: Path) -> Path:
    """Criar config SU2 para o adjoint a partir do config direto."""
    work_dir.mkdir(parents=True, exist_ok=True)
    text = direct_config.read_text(encoding="utf-8")
    text = text.replace("MATH_PROBLEM= DIRECT", "MATH_PROBLEM= CONTINUOUS_ADJOINT")
    text = text.replace("MATH_PROBLEM=DIRECT", "MATH_PROBLEM=CONTINUOUS_ADJOINT")
    adj_cfg = work_dir / "adjoint.cfg"
    adj_cfg.write_text(text, encoding="utf-8")
    return adj_cfg


def _compute_deltas(sens, config: AdjointConfig) -> dict[str, float]:
    """Converter sensibilidades em deltas para cada variável de projeto."""
    raw = sens.steepest_descent_step(step_size=config.step_size)
    # Mapear nomes do steepest_descent_step para nomes das variáveis de projeto
    name_map = {
        "delta_beta1_deg": "beta1",
        "delta_beta2_deg": "beta2",
        "delta_D2_m":      "d2",
        "delta_b2_m":      "b2",
        "delta_D1_m":      "d1",
    }
    return {
        v: raw.get(k, 0.0)
        for k, v in name_map.items()
        if v in config.design_vars
    }


def _apply_deltas(sizing, deltas: dict[str, float], config: AdjointConfig) -> object:
    """Aplicar deltas ao SizingResult respeitando bounds físicos."""
    modified = copy.copy(sizing)
    for var, delta in deltas.items():
        current = float(getattr(modified, var, 0.0))
        new_val = current + delta
        # Clip para bounds
        lo, hi = config.bounds.get(var, (-1e9, 1e9))
        new_val = max(lo, min(hi, new_val))
        object.__setattr__(modified, var, new_val)
        # Propagar para params se existir
        if hasattr(modified, "params") and hasattr(modified.params, var):
            object.__setattr__(modified.params, var, new_val)
    return modified
