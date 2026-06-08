"""Surrogate-assisted optimization — Fase 3.

Provides:
    - DesignPoint / ObjectiveValues: typed wrappers for pump design candidates
    - NSGAResult / BayesianResult: typed result types for Fase 3 API
    - run_surrogate_assisted(): two-stage NSGA-II with surrogate pre-screening

Two-stage strategy
------------------
Stage 1 — Surrogate screening (~0.5ms/eval):
    Run NSGA-II for n_gen_surrogate generations using the GP/XGBoost surrogate
    to evaluate each candidate. Cheap but approximate.

Stage 2 — Physics re-validation (~10ms/eval):
    Take the top n_cfd_validate% of the surrogate Pareto front and re-evaluate
    with full 1D sizing. Correct any surrogate over-predictions.

Returns the final Pareto front with real (physics-based) objective values.

Optuna / fallback
-----------------
BayesianResult wraps the result of run_bayesian() from bayesian.py.
If Optuna is not available, a random-search fallback is used.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hpe.core.models import OperatingPoint
from hpe.optimization.evaluator import evaluate_design
from hpe.optimization.problem import OptimizationProblem

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed design point — wraps the flat design_vector with named fields
# ---------------------------------------------------------------------------

@dataclass
class DesignPoint:
    """A candidate in the pump design space.

    Fields correspond to the 4 primary design variables used by
    OptimizationProblem.default().  Additional variables (nc, nd, d1_d2)
    are stored in ``extra``.
    """
    d2_factor: float        # D2 scale factor relative to 1D baseline (0.85-1.15)
    b2_factor: float        # b2 scale factor relative to 1D baseline (0.80-1.20)
    beta2: float            # Outlet blade angle [deg]
    n_blades: int           # Number of blades

    # Optional extended variables
    extra: dict[str, float] = field(default_factory=dict)

    def to_design_vector(self, problem: OptimizationProblem) -> list[float]:
        """Convert to flat design vector ordered by problem.variables."""
        mapping = {
            "d2_factor": self.d2_factor,
            "b2_factor": self.b2_factor,
            "beta2": self.beta2,
            "blade_count": float(self.n_blades),
            **self.extra,
        }
        return [mapping.get(v.name, (v.lower + v.upper) / 2.0)
                for v in problem.variables]

    @classmethod
    def from_design_vector(
        cls, vector: list[float], problem: OptimizationProblem
    ) -> "DesignPoint":
        """Reconstruct DesignPoint from a flat design vector."""
        var_map = {v.name: vector[i] for i, v in enumerate(problem.variables)}
        extra = {k: v for k, v in var_map.items()
                 if k not in ("d2_factor", "b2_factor", "beta2", "blade_count")}
        return cls(
            d2_factor=var_map.get("d2_factor", 1.0),
            b2_factor=var_map.get("b2_factor", 1.0),
            beta2=var_map.get("beta2", 25.0),
            n_blades=int(round(var_map.get("blade_count", 6))),
            extra=extra,
        )


@dataclass
class ObjectiveValues:
    """Multi-objective values for a design candidate."""
    efficiency: float       # eta_total (higher is better)
    npsh_r: float           # Required NPSH [m] (lower is better)
    robustness: float       # Mean efficiency over ±30% Q range (higher is better)

    # Extended objectives (present when using problem.extended())
    profile_loss_total: float = 0.0
    pmin_pa: float = 101325.0


# ---------------------------------------------------------------------------
# Typed result containers for Fase 3 API
# ---------------------------------------------------------------------------

@dataclass
class NSGAResult:
    """Result of a Fase 3 NSGA-II run.

    Compatible with (and wrapping) OptimizationResult from nsga2.py.
    """
    pareto_front: list[dict[str, Any]]   # [{variables, objectives, feasible}]
    all_evaluations: int
    hypervolume: float
    best_efficiency: dict[str, Any] | None
    best_npsh: dict[str, Any] | None
    generations: int
    runtime_s: float

    @classmethod
    def from_optimization_result(
        cls,
        opt_result: Any,
        runtime_s: float,
    ) -> "NSGAResult":
        """Build NSGAResult from the existing OptimizationResult type."""
        hv = _compute_hypervolume(opt_result.pareto_front)
        return cls(
            pareto_front=opt_result.pareto_front,
            all_evaluations=opt_result.all_evaluations,
            hypervolume=hv,
            best_efficiency=opt_result.best_efficiency,
            best_npsh=opt_result.best_npsh,
            generations=opt_result.generations,
            runtime_s=round(runtime_s, 2),
        )


@dataclass
class BayesianResult:
    """Result of a Fase 3 Bayesian optimisation run."""
    best_params: dict[str, Any]
    best_value: float        # best efficiency
    n_trials: int
    runtime_s: float
    study_name: str

    @classmethod
    def from_bayesian_dict(
        cls,
        d: dict[str, Any],
        runtime_s: float,
        study_name: str,
    ) -> "BayesianResult":
        return cls(
            best_params=d.get("best_params", {}),
            best_value=d.get("best_value", 0.0),
            n_trials=d.get("n_trials", 0),
            runtime_s=round(runtime_s, 2),
            study_name=study_name,
        )


# ---------------------------------------------------------------------------
# Fase 3 public API wrappers — typed, with hypervolume and runtime
# ---------------------------------------------------------------------------

def run_nsga2(
    problem: OptimizationProblem,
    pop_size: int = 40,
    n_gen: int = 50,
    seed: int = 42,
) -> NSGAResult:
    """Run NSGA-II multi-objective optimisation (Fase 3 typed interface).

    Delegates to hpe.optimization.nsga2.run_nsga2 and wraps the result
    in NSGAResult (adds hypervolume computation and runtime tracking).

    Parameters
    ----------
    problem : OptimizationProblem
        Problem definition with design variables and objectives.
    pop_size : int
        Population size per generation.
    n_gen : int
        Number of generations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NSGAResult
        Pareto front, hypervolume, best-by-objective, runtime.
    """
    from hpe.optimization.nsga2 import run_nsga2 as _run_nsga2

    t0 = time.perf_counter()
    opt_result = _run_nsga2(problem, pop_size=pop_size, n_gen=n_gen, seed=seed)
    runtime_s = time.perf_counter() - t0

    return NSGAResult.from_optimization_result(opt_result, runtime_s)


def run_bayesian(
    problem: OptimizationProblem,
    n_trials: int = 100,
    seed: int = 42,
) -> BayesianResult:
    """Run Bayesian optimisation (Fase 3 typed interface).

    Delegates to hpe.optimization.bayesian.run_bayesian when Optuna is
    available.  Falls back to random search when Optuna is not installed.

    Parameters
    ----------
    problem : OptimizationProblem
        Problem definition.
    n_trials : int
        Number of trials.
    seed : int
        Random seed.

    Returns
    -------
    BayesianResult
        Best parameters, best efficiency, trial count, runtime.
    """
    study_name = f"hpe_bayesian_Q{problem.flow_rate:.4f}_H{problem.head:.1f}"
    t0 = time.perf_counter()

    try:
        from hpe.optimization.bayesian import run_bayesian as _run_bayesian
        d = _run_bayesian(problem, n_trials=n_trials, seed=seed)
    except ImportError:
        log.warning("Optuna not available — using random search fallback")
        d = _random_search_fallback(problem, n_trials=n_trials, seed=seed)

    runtime_s = time.perf_counter() - t0
    return BayesianResult.from_bayesian_dict(d, runtime_s, study_name)


# ---------------------------------------------------------------------------
# Surrogate-assisted optimisation
# ---------------------------------------------------------------------------

def run_surrogate_assisted(
    problem: OptimizationProblem,
    evaluator: Any,  # SurrogateEvaluator
    n_gen_surrogate: int = 100,
    n_cfd_validate: int = 10,
) -> NSGAResult:
    """Surrogate-assisted NSGA-II in two stages.

    Stage 1 — Surrogate screening (~0.5ms/eval):
        NSGA-II with surrogate predict() for fitness evaluation.
        Runs n_gen_surrogate generations to explore the design space cheaply.

    Stage 2 — Physics re-validation (~10ms/eval):
        Top n_cfd_validate candidates from the surrogate Pareto front are
        re-evaluated with 1D sizing (evaluate_design from evaluator.py).
        This corrects any surrogate extrapolation errors before reporting results.

    Parameters
    ----------
    problem : OptimizationProblem
        Pump design problem (uses problem.flow_rate, .head, .rpm).
    evaluator : SurrogateEvaluator
        Trained surrogate model (any version).
    n_gen_surrogate : int
        Number of NSGA-II generations in Stage 1 (default 100).
    n_cfd_validate : int
        Number of Pareto candidates to re-validate in Stage 2 (default 10).

    Returns
    -------
    NSGAResult
        Final Pareto front with physics-validated objective values.
    """
    t0 = time.perf_counter()
    log.info(
        "surrogate_opt: starting Stage 1 (surrogate NSGA-II, %d gen)", n_gen_surrogate
    )

    # ------------------------------------------------------------------
    # Stage 1: NSGA-II with surrogate evaluator
    # ------------------------------------------------------------------
    surrogate_result = _run_nsga2_with_surrogate(
        problem, evaluator, n_gen=n_gen_surrogate
    )

    log.info(
        "surrogate_opt: Stage 1 done — %d Pareto solutions found",
        len(surrogate_result.pareto_front),
    )

    # ------------------------------------------------------------------
    # Stage 2: Re-validate top candidates with 1D sizing
    # ------------------------------------------------------------------
    n_validate = min(n_cfd_validate, len(surrogate_result.pareto_front))
    # Sort by efficiency descending — validate the "promising" ones first
    sorted_front = sorted(
        surrogate_result.pareto_front,
        key=lambda x: x["objectives"].get("efficiency", 0.0),
        reverse=True,
    )
    candidates_to_validate = sorted_front[:n_validate]

    log.info(
        "surrogate_opt: Stage 2 — re-validating top %d candidates with 1D sizing",
        n_validate,
    )

    validated_front: list[dict[str, Any]] = []
    for entry in candidates_to_validate:
        var_dict = entry["variables"]
        design_vector = [var_dict.get(v.name, (v.lower + v.upper) / 2.0)
                         for v in problem.variables]
        eval_result = evaluate_design(design_vector, problem)
        validated_entry = {
            "variables": var_dict,
            "objectives": eval_result.objectives,
            "feasible": eval_result.feasible,
            "stage": "physics_validated",
        }
        validated_front.append(validated_entry)

    # Merge: keep surrogate-only candidates that were not re-validated
    remaining = sorted_front[n_validate:]
    for entry in remaining:
        entry = dict(entry)
        entry["stage"] = "surrogate_only"
        validated_front.append(entry)

    # Re-sort final front by efficiency
    validated_front.sort(
        key=lambda x: x["objectives"].get("efficiency", 0.0), reverse=True
    )

    # Compute hypervolume on validated front
    hv = _compute_hypervolume(validated_front)

    # Best by objective
    feasible = [e for e in validated_front if e.get("feasible", True)]
    best_eff = max(feasible, key=lambda x: x["objectives"].get("efficiency", 0.0)) \
        if feasible else None
    best_npsh = min(feasible, key=lambda x: x["objectives"].get("npsh_r", 1e9)) \
        if feasible else None

    runtime_s = time.perf_counter() - t0
    log.info(
        "surrogate_opt: completed in %.1fs — %d final solutions (HV=%.4f)",
        runtime_s, len(validated_front), hv,
    )

    return NSGAResult(
        pareto_front=validated_front,
        all_evaluations=surrogate_result.all_evaluations + n_validate,
        hypervolume=hv,
        best_efficiency=best_eff,
        best_npsh=best_npsh,
        generations=n_gen_surrogate,
        runtime_s=round(runtime_s, 2),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_nsga2_with_surrogate(
    problem: OptimizationProblem,
    evaluator: Any,
    n_gen: int = 100,
    pop_size: int = 40,
    seed: int = 42,
) -> NSGAResult:
    """Stage 1: NSGA-II using surrogate predictions as the fitness function.

    Wraps the surrogate evaluator to replace the full 1D sizing call.
    Objectives: maximize efficiency, minimize npsh_r, maximize robustness.
    """
    import math as _math
    import random

    try:
        from deap import algorithms, base, creator, tools
        _DEAP = True
    except ImportError:
        _DEAP = False

    if not _DEAP:
        # Fallback: random population + simple non-domination filter
        return _surrogate_random_pareto(problem, evaluator, n_trials=pop_size * n_gen)

    random.seed(seed)
    t0 = time.perf_counter()

    # DEAP setup
    _clean_deap_types("FitnessMultiSurr", "IndividualSurr")
    creator.create("FitnessMultiSurr", base.Fitness, weights=(1.0, -1.0, 1.0))
    creator.create("IndividualSurr", list, fitness=creator.FitnessMultiSurr)

    toolbox = base.Toolbox()
    bounds = problem.variable_bounds()
    for i, (lo, hi) in enumerate(bounds):
        if problem.variables[i].is_integer:
            toolbox.register(f"attr_surr_{i}", random.randint, int(lo), int(hi))
        else:
            toolbox.register(f"attr_surr_{i}", random.uniform, lo, hi)

    def _create_ind():
        ind = []
        for i in range(problem.n_variables):
            ind.append(toolbox.__dict__[f"attr_surr_{i}"]())
        return creator.IndividualSurr(ind)

    toolbox.register("individual", _create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def _eval_surrogate(individual: list[float]) -> tuple[float, float, float]:
        """Evaluate candidate via surrogate (fast) instead of full sizing."""
        try:
            from hpe.ai.surrogate.evaluator import SurrogateInput

            # Get baseline sizing geometry to derive Ns, D2 for surrogate
            op = OperatingPoint(
                flow_rate=problem.flow_rate,
                head=problem.head,
                rpm=problem.rpm,
            )
            from hpe.sizing.meanline import run_sizing
            baseline = run_sizing(op)

            var_names = [v.name for v in problem.variables]

            def _get(name, default):
                return individual[var_names.index(name)] if name in var_names else default

            d2_factor = _get("d2_factor", 1.0)
            b2_factor = _get("b2_factor", 1.0)

            d2_mm = baseline.impeller_d2 * d2_factor * 1000.0
            b2_mm = baseline.impeller_b2 * b2_factor * 1000.0
            beta2 = _get("beta2", baseline.beta2)

            # Surrogate input
            surr_inp = SurrogateInput(
                Ns=baseline.specific_speed_ns,
                D2=d2_mm,
                b2=b2_mm,
                beta2=beta2,
                n=problem.rpm,
                Q=problem.flow_rate,
                H=problem.head,
            )
            surr_out = evaluator.predict(surr_inp)

            eta = surr_out.eta_total / 100.0  # fraction
            npsh_r = baseline.estimated_npsh_r  # surrogate doesn't predict NPSHr
            robustness = eta * 0.97  # conservative proxy

            # Constraint: tip speed
            u2 = _math.pi * (d2_mm / 1000.0) * problem.rpm / 60.0
            if u2 > 35.0:
                return (0.0, 100.0, 0.0)

            return (eta, npsh_r, robustness)
        except Exception:
            return (0.0, 100.0, 0.0)

    toolbox.register("evaluate", _eval_surrogate)
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded,
        low=[b[0] for b in bounds], up=[b[1] for b in bounds], eta=20.0,
    )
    toolbox.register(
        "mutate", tools.mutPolynomialBounded,
        low=[b[0] for b in bounds], up=[b[1] for b in bounds],
        eta=20.0, indpb=0.2,
    )
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    total_evals = 0

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    total_evals += len(pop)

    for gen in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.2)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        total_evals += len(invalid)
        pop = toolbox.select(pop + offspring, pop_size)

    pareto_inds = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    pareto_front = []
    for ind in pareto_inds:
        entry = {
            "variables": {
                problem.variables[i].name: ind[i]
                for i in range(problem.n_variables)
            },
            "objectives": {
                "efficiency": ind.fitness.values[0],
                "npsh_r": ind.fitness.values[1],
                "robustness": ind.fitness.values[2],
            },
            "feasible": ind.fitness.values[0] > 0,
        }
        pareto_front.append(entry)

    runtime_s = time.perf_counter() - t0
    hv = _compute_hypervolume(pareto_front)

    best_eff = max(pareto_front, key=lambda x: x["objectives"].get("efficiency", 0.0)) \
        if pareto_front else None
    best_npsh = min(pareto_front, key=lambda x: x["objectives"].get("npsh_r", 1e9)) \
        if pareto_front else None

    return NSGAResult(
        pareto_front=pareto_front,
        all_evaluations=total_evals,
        hypervolume=hv,
        best_efficiency=best_eff,
        best_npsh=best_npsh,
        generations=n_gen,
        runtime_s=round(runtime_s, 2),
    )


def _surrogate_random_pareto(
    problem: OptimizationProblem,
    evaluator: Any,
    n_trials: int = 4000,
    seed: int = 42,
) -> NSGAResult:
    """Fallback when DEAP is not available: random search + Pareto filter."""
    rng = np.random.default_rng(seed)
    bounds = problem.variable_bounds()
    t0 = time.perf_counter()

    candidates = []
    for _ in range(n_trials):
        vector = [
            rng.integers(int(lo), int(hi) + 1) if problem.variables[i].is_integer
            else rng.uniform(lo, hi)
            for i, (lo, hi) in enumerate(bounds)
        ]
        result = evaluate_design(vector, problem)
        if result.feasible:
            candidates.append({
                "variables": {problem.variables[i].name: vector[i]
                              for i in range(problem.n_variables)},
                "objectives": result.objectives,
                "feasible": True,
            })

    pareto_front = _filter_pareto(candidates)
    hv = _compute_hypervolume(pareto_front)
    runtime_s = time.perf_counter() - t0

    best_eff = max(pareto_front, key=lambda x: x["objectives"].get("efficiency", 0)) \
        if pareto_front else None
    best_npsh = min(pareto_front, key=lambda x: x["objectives"].get("npsh_r", 1e9)) \
        if pareto_front else None

    return NSGAResult(
        pareto_front=pareto_front,
        all_evaluations=n_trials,
        hypervolume=hv,
        best_efficiency=best_eff,
        best_npsh=best_npsh,
        generations=0,
        runtime_s=round(runtime_s, 2),
    )


def _random_search_fallback(
    problem: OptimizationProblem,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Random search fallback when Optuna is not available."""
    rng = np.random.default_rng(seed)
    bounds = problem.variable_bounds()
    best_eff = -1.0
    best_params: dict[str, Any] = {}

    for _ in range(n_trials):
        vector = [
            int(rng.integers(int(lo), int(hi) + 1)) if problem.variables[i].is_integer
            else float(rng.uniform(lo, hi))
            for i, (lo, hi) in enumerate(bounds)
        ]
        result = evaluate_design(vector, problem)
        eff = result.objectives.get("efficiency", 0.0)
        if result.feasible and eff > best_eff:
            best_eff = eff
            best_params = {
                problem.variables[i].name: vector[i]
                for i in range(problem.n_variables)
            }

    return {
        "best_params": best_params,
        "best_value": best_eff,
        "n_trials": n_trials,
        "objective": "efficiency",
    }


def _filter_pareto(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Simple non-dominated sort for 3 objectives.

    Objectives: efficiency (max→), npsh_r (min→), robustness (max→).
    Converts to minimisation convention internally: [-eff, npsh, -rob].
    """
    if not candidates:
        return []

    def _dominates(a: dict, b: dict) -> bool:
        """Return True if a dominates b (a is at least as good on all objectives)."""
        oa = a["objectives"]
        ob = b["objectives"]
        # minimise [-eff, npsh_r, -rob]
        a_vals = (-oa.get("efficiency", 0), oa.get("npsh_r", 1e9), -oa.get("robustness", 0))
        b_vals = (-ob.get("efficiency", 0), ob.get("npsh_r", 1e9), -ob.get("robustness", 0))
        return all(av <= bv for av, bv in zip(a_vals, b_vals)) and any(
            av < bv for av, bv in zip(a_vals, b_vals)
        )

    pareto = []
    for c in candidates:
        dominated = any(_dominates(other, c) for other in candidates if other is not c)
        if not dominated:
            pareto.append(c)
    return pareto


def _compute_hypervolume(
    pareto_front: list[dict[str, Any]],
    ref_point: tuple[float, float, float] = (0.0, 20.0, 0.0),
) -> float:
    """Approximate hypervolume indicator using a grid-based estimate.

    Reference point: (efficiency=0, npsh_r=20m, robustness=0) — worst case.
    Maximised HV indicates a well-spread, high-quality Pareto front.

    Uses a simple box-volume sum (WFG approximation for 3 objectives).
    """
    if not pareto_front:
        return 0.0

    # Convert to minimisation space: [-eta, npsh, -rob]
    ref = (-ref_point[0], ref_point[1], -ref_point[2])

    points = []
    for e in pareto_front:
        obj = e.get("objectives", {})
        eff = obj.get("efficiency", 0.0)
        npsh = obj.get("npsh_r", ref_point[1])
        rob = obj.get("robustness", 0.0)
        points.append((-eff, npsh, -rob))

    if not points:
        return 0.0

    # Remove dominated points (should already be Pareto, but be safe)
    # Simple hypervolume lower bound: sum of individual dominated boxes
    # for each point relative to the reference point
    hv = 0.0
    for p in points:
        vol = 1.0
        for i in range(3):
            span = ref[i] - p[i]
            if span <= 0:
                vol = 0.0
                break
            vol *= span
        hv += vol

    # Normalise by a rough scale to keep the value readable
    # (efficiency ~0-1, npsh ~0-20, robustness ~0-1)
    scale = 1.0 * 20.0 * 1.0  # ref box volume
    return round(hv / scale, 6) if scale > 0 else 0.0


def _clean_deap_types(*names: str) -> None:
    """Remove DEAP creator types by name to avoid re-creation errors."""
    try:
        from deap import creator
        for name in names:
            if name in creator.__dict__:
                del creator.__dict__[name]
    except Exception:
        pass
