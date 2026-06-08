"""Melhorias de otimização — #26-30.

- crowding_distance: NSGA-II proper
- ConstraintHandler: penalty + repair
- OptimizationCheckpoint: salvar/restaurar estado
- ActiveLearningLoop: query by uncertainty
- SurrogateAutoTune: hyperparameter search
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ===========================================================================
# #26 Crowding distance (NSGA-II proper)
# ===========================================================================

def crowding_distance(
    population: list[dict],
    objectives: list[str],
) -> list[float]:
    """Calcular crowding distance NSGA-II para diversidade.

    Para cada indivíduo, soma normalizada das distâncias aos vizinhos
    em cada objetivo. Pontos no limite recebem ∞ (preservados).
    """
    n = len(population)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    for obj in objectives:
        # Sort indices by this objective
        indices = sorted(range(n), key=lambda i: population[i].get(obj, 0))
        f_min = population[indices[0]].get(obj, 0)
        f_max = population[indices[-1]].get(obj, 0)
        rng = f_max - f_min
        if rng == 0:
            continue

        distances[indices[0]] = float("inf")
        distances[indices[-1]] = float("inf")
        for k in range(1, n - 1):
            d = (population[indices[k + 1]].get(obj, 0)
                 - population[indices[k - 1]].get(obj, 0)) / rng
            distances[indices[k]] += d

    return distances


# ===========================================================================
# #27 Constraint handling
# ===========================================================================

@dataclass
class Constraint:
    name: str
    fn: Callable[[dict], float]   # g(x) ≤ 0 means feasible
    weight: float = 100.0
    type: str = "inequality"      # 'inequality' | 'equality'


class ConstraintHandler:
    """Penalty + repair para problemas com restrições.

    Penalty: f_penalized = f + Σ w_i × max(0, g_i(x))²
    Repair: tenta ajustar x para satisfazer constraints (clipping de
            variáveis para bounds, projeção em manifolds simples).
    """

    def __init__(self, constraints: Optional[list[Constraint]] = None):
        self.constraints = constraints or []

    def evaluate(self, x: dict, fitness: float) -> tuple[float, list[float]]:
        """Avaliar fitness penalizado + lista de violações."""
        violations = []
        penalty = 0.0
        for c in self.constraints:
            try:
                g = c.fn(x)
                violations.append(g)
                if c.type == "inequality":
                    penalty += c.weight * max(0.0, g) ** 2
                else:
                    penalty += c.weight * g ** 2
            except Exception as exc:
                log.warning("Constraint %s failed: %s", c.name, exc)
                violations.append(float("nan"))

        return fitness + penalty, violations

    def repair(self, x: dict, bounds: dict[str, tuple[float, float]]) -> dict:
        """Reparo simples por clipping nos bounds."""
        out = dict(x)
        for k, (lo, hi) in bounds.items():
            if k in out:
                out[k] = max(lo, min(hi, out[k]))
        return out

    def is_feasible(self, x: dict) -> bool:
        for c in self.constraints:
            try:
                if c.type == "inequality" and c.fn(x) > 1e-6:
                    return False
                if c.type == "equality" and abs(c.fn(x)) > 1e-6:
                    return False
            except Exception:
                return False
        return True


# ===========================================================================
# #28 Optimization checkpoint/restart
# ===========================================================================

@dataclass
class OptimizationCheckpoint:
    """Estado serializável de uma execução de otimização."""
    run_id: str
    iteration: int
    population: list[dict] = field(default_factory=list)
    best_fitness: float = 0.0
    best_individual: dict = field(default_factory=dict)
    rng_state: Optional[list] = None
    metadata: dict = field(default_factory=dict)

    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "run_id": self.run_id,
            "iteration": self.iteration,
            "population": self.population,
            "best_fitness": self.best_fitness,
            "best_individual": self.best_individual,
            "rng_state": self.rng_state,
            "metadata": self.metadata,
        }, indent=2))
        log.info("Checkpoint saved: %s (iter %d)", path, self.iteration)

    @classmethod
    def load(cls, path: "str | Path") -> "OptimizationCheckpoint":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def restore_rng(self) -> random.Random:
        rng = random.Random()
        if self.rng_state:
            rng.setstate(tuple(self.rng_state) if isinstance(self.rng_state, list)
                         else self.rng_state)
        return rng


# ===========================================================================
# #29 Active learning loop (query by uncertainty)
# ===========================================================================

@dataclass
class ActiveLearningQuery:
    """Resultado de uma rodada de active learning."""
    candidates: list[dict]
    selected: list[dict]
    selection_scores: list[float]
    n_iterations_total: int
    avg_uncertainty: float


def active_learning_query(
    candidate_pool: list[dict],
    surrogate_predict_uncertainty: Callable[[dict], tuple[float, float]],
    n_select: int = 5,
    strategy: str = "max_uncertainty",
) -> ActiveLearningQuery:
    """Selecionar próximos pontos para CFD via critério de incerteza.

    strategy:
      'max_uncertainty': escolhe N maiores σ
      'expected_improvement': EI = (μ - f_best) × Φ + σ × φ
      'random': baseline para comparação
    """
    if not candidate_pool:
        return ActiveLearningQuery([], [], [], 0, 0.0)

    scored: list[tuple[dict, float, float]] = []  # (x, mean, std)
    for x in candidate_pool:
        try:
            mu, sigma = surrogate_predict_uncertainty(x)
            scored.append((x, mu, sigma))
        except Exception as exc:
            log.warning("Active learning predict failed: %s", exc)

    if strategy == "random":
        rng = random.Random(42)
        selected = rng.sample(scored, min(n_select, len(scored)))
        scores = [s[2] for s in selected]
    elif strategy == "expected_improvement":
        f_best = min(s[1] for s in scored)
        # Simplified EI
        eis = []
        for x, mu, sigma in scored:
            improvement = max(0, f_best - mu)
            ei = improvement + 0.5 * sigma
            eis.append((x, ei, sigma))
        eis.sort(key=lambda t: -t[1])
        selected = [(t[0], 0.0, t[2]) for t in eis[:n_select]]
        scores = [t[1] for t in eis[:n_select]]
    else:
        # max_uncertainty
        scored.sort(key=lambda s: -s[2])
        selected = scored[:n_select]
        scores = [s[2] for s in selected]

    avg_unc = sum(s[2] for s in scored) / len(scored)

    return ActiveLearningQuery(
        candidates=[s[0] for s in scored],
        selected=[s[0] for s in selected],
        selection_scores=scores,
        n_iterations_total=len(scored),
        avg_uncertainty=avg_unc,
    )


# ===========================================================================
# #30 Surrogate hyperparameter auto-tune
# ===========================================================================

@dataclass
class HyperparameterSearchResult:
    best_params: dict
    best_score: float
    n_trials: int
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6),
            "n_trials": self.n_trials,
            "history_size": len(self.history),
        }


def auto_tune_surrogate(
    score_fn: Callable[[dict], float],
    param_space: dict[str, list],
    n_trials: int = 20,
    seed: int = 42,
) -> HyperparameterSearchResult:
    """Random search simples (Bayesian opt seria via Optuna).

    param_space: {"max_depth": [3,5,7], "lr": [0.01, 0.05, 0.1], ...}
    score_fn: recebe dict de hyperparams, retorna score (maior=melhor)
    """
    rng = random.Random(seed)
    history = []
    best_params = {}
    best_score = -float("inf")

    for trial in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        try:
            score = score_fn(params)
        except Exception as exc:
            log.warning("Trial %d failed: %s", trial, exc)
            score = -float("inf")
        history.append({"trial": trial, "params": params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params

    return HyperparameterSearchResult(
        best_params=best_params,
        best_score=best_score,
        n_trials=n_trials,
        history=history,
    )
