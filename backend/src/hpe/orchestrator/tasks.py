"""Celery tasks for HPE pipeline stages.

Each task wraps a specific HPE module and returns a JSON-serialisable dict.
Tasks can be chained:  sizing → geometry → surrogate → cfd.

Queues
------
  fast      sizing, geometry, surrogate  (concurrency 8, retries 3)
  cfd       OpenFOAM simulation          (concurrency 2, retries 1)
  optimize  NSGA-II / Optuna             (concurrency 4, retries 2)

Fallback
--------
If Celery is not installed the tasks are still importable and can be called
synchronously via ``task_fn(arg)`` instead of ``task_fn.delay(arg)``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Celery import — keep module importable even without celery installed
# ---------------------------------------------------------------------------
try:
    from hpe.orchestrator.config import celery_app as _celery_app

    _CELERY_AVAILABLE = _celery_app is not None
except Exception:  # pragma: no cover
    _celery_app = None
    _CELERY_AVAILABLE = False


class _FakeTask:
    """Synchronous task shim used when Celery is not installed.

    Exposes ``.delay()`` / ``.apply_async()`` that call the wrapped function
    immediately in the current thread.  When ``bind=True`` was declared, a
    lightweight self-stub is injected so the function receives the expected
    first positional argument.
    """

    def __init__(self, fn, bound: bool = False):
        import functools
        self._fn = fn
        self._bound = bound
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if self._bound:
            return self._fn(self, *args, **kwargs)
        return self._fn(*args, **kwargs)

    class _FakeResult:
        def __init__(self, value):
            self._value = value
            self.id = "sync-task"

        def get(self, timeout=None):
            return self._value

    # Celery-compatible attributes
    def update_state(self, state=None, meta=None):
        pass  # no-op in synchronous mode

    def delay(self, *args, **kwargs):
        return self._FakeResult(self(*args, **kwargs))

    def apply_async(self, args=(), kwargs=None, **_):
        return self._FakeResult(self(*list(args), **(kwargs or {})))

    def s(self, *args, **kwargs):
        """Partial signature stub (for Celery canvas compatibility)."""
        return (self._fn, args, kwargs)


def _task(*args, **kwargs):
    """Decorator factory: uses @celery_app.task when Celery is available,
    otherwise wraps the function with a _FakeTask synchronous shim that
    exposes ``.delay()`` and ``.apply_async()``."""

    bound = kwargs.get("bind", False)

    def decorator(fn):
        if _CELERY_AVAILABLE and _celery_app is not None:
            return _celery_app.task(*args, **kwargs)(fn)

        return _FakeTask(fn, bound=bound)

    return decorator


# ---------------------------------------------------------------------------
# Helper: OperatingPoint serialisation
# ---------------------------------------------------------------------------

def _op_from_dict(op_dict: dict) -> Any:
    """Deserialise OperatingPoint from a plain dict.

    Accepts both camelCase / upper keys (Q, H, n) and snake_case
    (flow_rate, head, rpm).
    """
    from hpe.core.models import OperatingPoint

    # Normalise keys
    Q = op_dict.get("flow_rate") or op_dict.get("Q") or op_dict.get("q_m3h", 0)
    H = op_dict.get("head") or op_dict.get("H") or op_dict.get("h_m", 0)
    n = op_dict.get("rpm") or op_dict.get("n") or op_dict.get("n_rpm", 1450)
    fluid = op_dict.get("fluid_type")
    pre_swirl = op_dict.get("pre_swirl_angle", 0.0)

    # Convert Q from m3/h to m3/s if it appears to be in m3/h (>1 typically)
    if Q > 1.0:
        Q = Q / 3600.0

    kwargs: dict[str, Any] = {"flow_rate": float(Q), "head": float(H), "rpm": float(n)}
    if fluid:
        kwargs["fluid_type"] = fluid
    if pre_swirl:
        kwargs["pre_swirl_angle"] = float(pre_swirl)

    return OperatingPoint(**kwargs)


def _sizing_to_dict(sizing) -> dict:
    """Serialise SizingResult to a JSON-safe dict."""
    return {
        "specific_speed_ns": sizing.specific_speed_ns,
        "specific_speed_nq": sizing.specific_speed_nq,
        "impeller_d2": sizing.impeller_d2,
        "impeller_d1": sizing.impeller_d1,
        "impeller_b2": sizing.impeller_b2,
        "blade_count": sizing.blade_count,
        "beta1": sizing.beta1,
        "beta2": sizing.beta2,
        "estimated_efficiency": sizing.estimated_efficiency,
        "estimated_power": sizing.estimated_power,
        "estimated_npsh_r": getattr(sizing, "estimated_npsh_r", None),
        "sigma": getattr(sizing, "sigma", None),
        "diffusion_ratio": getattr(sizing, "diffusion_ratio", 0.0),
        "warnings": list(sizing.warnings),
    }


# ---------------------------------------------------------------------------
# Task: run_sizing
# ---------------------------------------------------------------------------

@_task(name="hpe.tasks.run_sizing", queue="fast", bind=True)
def run_sizing_task(self, op_dict: dict) -> dict:
    """Execute 1D meanline sizing and return a JSON-serialisable dict.

    Parameters
    ----------
    op_dict : dict
        Operating point.  Accepted keys: ``Q`` (m³/h), ``H`` (m), ``n`` (rpm)
        *or* ``flow_rate`` (m³/s), ``head``, ``rpm``.

    Returns
    -------
    dict
        SizingResult serialised including D2, efficiency, warnings, etc.
    """
    log.info("run_sizing_task: start op=%s", op_dict)
    t0 = time.perf_counter()

    try:
        from hpe.sizing.meanline import run_sizing

        if _CELERY_AVAILABLE and hasattr(self, "update_state"):
            self.update_state(state="RUNNING", meta={"step": "sizing", "progress": 10})

        op = _op_from_dict(op_dict)
        sizing = run_sizing(op)
        result = _sizing_to_dict(sizing)
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        log.info(
            "run_sizing_task: done D2=%.1fmm eta=%.3f elapsed=%.1fms",
            sizing.impeller_d2 * 1000,
            sizing.estimated_efficiency,
            result["elapsed_ms"],
        )
        return result

    except Exception as exc:
        log.exception("run_sizing_task: failed — %s", exc)
        raise RuntimeError(f"Sizing failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Task: run_geometry
# ---------------------------------------------------------------------------

@_task(name="hpe.tasks.run_geometry", queue="fast", bind=True)
def run_geometry_task(self, sizing_dict: dict) -> dict:
    """Generate parametric runner geometry from a serialised SizingResult.

    Parameters
    ----------
    sizing_dict : dict
        Output of ``run_sizing_task`` (contains impeller_d2, beta2, etc.).

    Returns
    -------
    dict
        GeometryResult serialised via ``to_dict()``.
    """
    log.info("run_geometry_task: start")
    t0 = time.perf_counter()

    try:
        from hpe.geometry.parametric import run_geometry
        from hpe.core.models import SizingResult

        if _CELERY_AVAILABLE and hasattr(self, "update_state"):
            self.update_state(state="RUNNING", meta={"step": "geometry", "progress": 30})

        # Re-hydrate SizingResult from dict
        sr = SizingResult(
            specific_speed_ns=sizing_dict.get("specific_speed_ns", 0.0),
            specific_speed_nq=sizing_dict.get("specific_speed_nq", 0.0),
            impeller_d2=sizing_dict.get("impeller_d2", 0.0),
            impeller_d1=sizing_dict.get("impeller_d1", 0.0),
            impeller_b2=sizing_dict.get("impeller_b2", 0.0),
            blade_count=int(sizing_dict.get("blade_count", 6)),
            beta1=sizing_dict.get("beta1", 20.0),
            beta2=sizing_dict.get("beta2", 25.0),
            estimated_efficiency=sizing_dict.get("estimated_efficiency", 0.8),
            estimated_power=sizing_dict.get("estimated_power", 0.0),
            estimated_npsh_r=sizing_dict.get("estimated_npsh_r", 0.0),
            sigma=sizing_dict.get("sigma", 0.0),
            warnings=sizing_dict.get("warnings", []),
        )

        geo = run_geometry(sr)
        result = geo.to_dict()
        result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["cad_available"] = geo.cad_available
        result["step_path"] = geo.step_path
        result["stl_path"] = geo.stl_path
        result["generation_time_ms"] = geo.generation_time_ms
        result["warnings"] = geo.warnings

        log.info(
            "run_geometry_task: done cad=%s elapsed=%.1fms",
            geo.cad_available,
            result["elapsed_ms"],
        )
        return result

    except Exception as exc:
        log.exception("run_geometry_task: failed — %s", exc)
        raise RuntimeError(f"Geometry failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Task: run_surrogate
# ---------------------------------------------------------------------------

@_task(name="hpe.tasks.run_surrogate", queue="fast", bind=True)
def run_surrogate_task(self, sizing_dict: dict) -> dict:
    """Run surrogate model prediction from sizing results.

    Parameters
    ----------
    sizing_dict : dict
        Output of ``run_sizing_task`` (or equivalent).

    Returns
    -------
    dict
        ``{"eta_total": float, "eta_hid": float, "p_kw": float,
           "confidence": float, "elapsed_ms": float}``
    """
    log.info("run_surrogate_task: start")
    t0 = time.perf_counter()

    try:
        from hpe.ai.surrogate.evaluator import SurrogateEvaluator, SurrogateInput

        if _CELERY_AVAILABLE and hasattr(self, "update_state"):
            self.update_state(state="RUNNING", meta={"step": "surrogate", "progress": 50})

        ev = SurrogateEvaluator.load_default()

        ns = sizing_dict.get("specific_speed_ns", 0.0)
        d2_mm = sizing_dict.get("impeller_d2", 0.0) * 1000
        b2_mm = sizing_dict.get("impeller_b2", 0.0) * 1000
        beta2 = sizing_dict.get("beta2", 25.0)
        n_rpm = sizing_dict.get("n_rpm", sizing_dict.get("rpm", 1450.0))
        Q = sizing_dict.get("Q", sizing_dict.get("flow_rate", 0.0))
        H = sizing_dict.get("H", sizing_dict.get("head", 0.0))
        # Q stored in m3/s in sizing; convert to m3/h for surrogate
        q_m3h = Q * 3600 if Q < 10 else Q

        inp = SurrogateInput(
            Ns=ns, D2=d2_mm, b2=b2_mm, beta2=beta2,
            n=n_rpm, Q=Q, H=H,
        )
        pred = ev.predict(inp)

        # SurrogateOutput fields: eta_hid, H, P_shaft, confidence, surrogate_version, latency_ms
        # Normalise to canonical API names for downstream consumers
        eta_hid_raw = getattr(pred, "eta_hid", None)
        p_shaft_raw = getattr(pred, "P_shaft", getattr(pred, "p_kw", None))
        conf_raw = getattr(pred, "confidence", None)
        h_raw = getattr(pred, "H", None)

        # eta_hid from surrogate is stored as percentage (e.g. 64.9) — convert to fraction
        eta_hid = round(float(eta_hid_raw) / 100.0, 4) if eta_hid_raw is not None else None
        # eta_total ≈ eta_hid × volumetric × mechanical (simplified: use eta_hid as proxy)
        eta_total = eta_hid

        result = {
            "eta_total": eta_total,
            "eta_hid": eta_hid,
            "p_kw": round(float(p_shaft_raw), 3) if p_shaft_raw is not None else None,
            "H_surrogate": round(float(h_raw), 2) if h_raw is not None else None,
            "confidence": round(float(conf_raw), 4) if conf_raw is not None else None,
            "surrogate_version": getattr(pred, "surrogate_version", "v1"),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        log.info(
            "run_surrogate_task: eta_total=%.3f confidence=%s",
            result["eta_total"],
            result["confidence"],
        )
        return result

    except Exception as exc:
        log.exception("run_surrogate_task: failed — %s", exc)
        # Surrogate failure is non-fatal — return degraded result
        return {
            "eta_total": None,
            "eta_hid": None,
            "p_kw": None,
            "confidence": None,
            "error": str(exc),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


# ---------------------------------------------------------------------------
# Task: run_cfd
# ---------------------------------------------------------------------------

@_task(
    name="hpe.tasks.run_cfd",
    queue="cfd",
    bind=True,
    soft_time_limit=3600,
    time_limit=4000,
)
def run_cfd_task(self, sizing_dict: dict, output_dir: str = "/tmp/hpe_cfd") -> dict:
    """Execute the full CFD pipeline for a given sizing result.

    Long-running (~30 min).  Updates Redis status via ``status.tracker``
    so the WebSocket endpoint can stream progress to the client.

    Parameters
    ----------
    sizing_dict : dict
        Serialised SizingResult (output of ``run_sizing_task``).
    output_dir : str
        Directory where CFD case files will be written.

    Returns
    -------
    dict
        CFD performance extraction results.
    """
    from hpe.orchestrator.status import tracker

    run_id = sizing_dict.get("_run_id", "cfd-unknown")
    log.info("run_cfd_task: start run_id=%s output_dir=%s", run_id, output_dir)
    t0 = time.perf_counter()

    def _update(progress: int, stage: str, msg: str = "") -> None:
        tracker.update_progress(run_id, progress, stage, msg)
        if _CELERY_AVAILABLE and hasattr(self, "update_state"):
            self.update_state(
                state="RUNNING",
                meta={"step": stage, "progress": progress, "message": msg},
            )

    try:
        _update(5, "cfd_init", "Initialising CFD case")

        # Try to import CFD pipeline; fall back to stub if not implemented yet
        try:
            from hpe.cfd.pipeline import run_cfd_pipeline  # type: ignore[import]
            _update(10, "cfd_mesh", "Generating mesh")
            result = run_cfd_pipeline(
                sizing_dict,
                output_dir=output_dir,
                progress_callback=_update,
            )
        except ImportError:
            log.warning("run_cfd_task: hpe.cfd.pipeline not yet implemented — returning stub")
            _update(100, "cfd_done", "CFD stub (pipeline not yet implemented)")
            result = {
                "status": "stub",
                "message": "CFD pipeline module not yet implemented (Fase 2+)",
                "eta_hid": None,
                "p_kw": None,
            }

        result["elapsed_s"] = round(time.perf_counter() - t0, 2)
        _update(100, "done", "CFD completed")
        log.info("run_cfd_task: done elapsed=%.1fs", result["elapsed_s"])
        return result

    except Exception as exc:
        log.exception("run_cfd_task: failed — %s", exc)
        tracker.update_progress(run_id, 0, "failed", str(exc))
        raise RuntimeError(f"CFD failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Task: run_optimization
# ---------------------------------------------------------------------------

@_task(name="hpe.tasks.run_optimization", queue="optimize", bind=True)
def run_optimization_task(
    self,
    op_dict: dict,
    method: str = "nsga2",
    pop_size: int = 40,
    n_gen: int = 50,
) -> dict:
    """Execute multi-objective optimisation (NSGA-II or Bayesian).

    Parameters
    ----------
    op_dict : dict
        Operating point dict.
    method : str
        ``"nsga2"`` or ``"bayesian"``.
    pop_size : int
        Population size (NSGA-II).
    n_gen : int
        Number of generations / trials.

    Returns
    -------
    dict
        ``{"pareto_front": [...], "n_evaluations": int, "elapsed_s": float}``
    """
    log.info("run_optimization_task: start method=%s pop=%d gen=%d", method, pop_size, n_gen)
    t0 = time.perf_counter()

    if _CELERY_AVAILABLE and hasattr(self, "update_state"):
        self.update_state(
            state="RUNNING",
            meta={"step": "optimization", "method": method, "progress": 0},
        )

    try:
        from hpe.optimization.problem import OptimizationProblem  # type: ignore[import]

        op = _op_from_dict(op_dict)

        problem = OptimizationProblem.default(op.flow_rate, op.head, op.rpm)

        if method == "nsga2":
            from hpe.optimization.nsga2 import run_nsga2  # type: ignore[import]
            result = run_nsga2(problem, pop_size=pop_size, n_gen=n_gen)
            return {
                "method": "nsga2",
                "pareto_front": result.pareto_front,
                "n_evaluations": result.all_evaluations,
                "best_efficiency": getattr(result, "best_efficiency", None),
                "best_npsh": getattr(result, "best_npsh", None),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
        elif method == "bayesian":
            from hpe.optimization.bayesian import run_bayesian  # type: ignore[import]
            result = run_bayesian(problem, n_trials=n_gen)
            return {
                "method": "bayesian",
                "pareto_front": [{"variables": result["best_params"],
                                  "objectives": {"efficiency": result["best_value"]}}],
                "n_evaluations": result.get("n_trials", n_gen),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
        else:
            raise ValueError(f"Unknown optimisation method: {method!r}")

    except Exception as exc:
        log.exception("run_optimization_task: failed — %s", exc)
        raise RuntimeError(f"Optimisation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Task: run_full_pipeline
# ---------------------------------------------------------------------------

@_task(name="hpe.tasks.run_full_pipeline", queue="fast", bind=True)
def run_full_pipeline_task(self, op_dict: dict, run_cfd: bool = False) -> dict:
    """Execute the full design pipeline: sizing → geometry → surrogate → (CFD optional).

    Parameters
    ----------
    op_dict : dict
        Operating point.
    run_cfd : bool
        If ``True``, also run the CFD stage after sizing (long-running).

    Returns
    -------
    dict
        PipelineResult with keys from all completed stages.
    """
    from hpe.orchestrator.status import tracker
    from hpe.orchestrator.versions import DesignVersion, save_version

    run_id = op_dict.get("_run_id", "pipeline-" + str(int(time.time())))
    log.info("run_full_pipeline_task: start run_id=%s run_cfd=%s", run_id, run_cfd)
    t0 = time.perf_counter()

    def _update(progress: int, stage: str, msg: str = "") -> None:
        tracker.update_progress(run_id, progress, stage, msg)
        if _CELERY_AVAILABLE and hasattr(self, "update_state"):
            self.update_state(
                state="RUNNING",
                meta={"step": stage, "progress": progress, "message": msg},
            )

    pipeline_result: dict[str, Any] = {"run_id": run_id, "status": "running"}

    try:
        # ── Stage 1: Sizing ───────────────────────────────────────────────
        _update(10, "sizing", "Running 1D meanline sizing")
        sizing_dict = run_sizing_task(op_dict)
        pipeline_result["sizing"] = sizing_dict
        log.info("pipeline sizing done: D2=%.1fmm", sizing_dict.get("impeller_d2", 0) * 1000)

        # ── Stage 2: Geometry ─────────────────────────────────────────────
        _update(35, "geometry", "Generating parametric geometry")
        # Pass through operating point metadata for geometry
        sizing_with_op = {**sizing_dict, **{
            "Q": op_dict.get("Q") or op_dict.get("flow_rate", 0),
            "H": op_dict.get("H") or op_dict.get("head", 0),
            "n_rpm": op_dict.get("n") or op_dict.get("rpm", 1450),
        }}
        geo_dict = run_geometry_task(sizing_with_op)
        pipeline_result["geometry"] = geo_dict
        log.info("pipeline geometry done: cad=%s", geo_dict.get("cad_available"))

        # ── Stage 3: Surrogate prediction ─────────────────────────────────
        _update(60, "surrogate", "Running surrogate prediction")
        surrogate_dict = run_surrogate_task(sizing_with_op)
        pipeline_result["surrogate"] = surrogate_dict
        log.info("pipeline surrogate done: eta_total=%s", surrogate_dict.get("eta_total"))

        # ── Stage 4 (optional): CFD ───────────────────────────────────────
        if run_cfd:
            _update(70, "cfd", "Submitting CFD case")
            cfd_sizing = {**sizing_dict, "_run_id": run_id}
            cfd_dict = run_cfd_task(cfd_sizing)
            pipeline_result["cfd"] = cfd_dict
            log.info("pipeline cfd done: status=%s", cfd_dict.get("status"))

        # ── Versioning ────────────────────────────────────────────────────
        _update(90, "versioning", "Saving design version")
        version = DesignVersion.from_sizing(
            op_dict=op_dict,
            sizing_dict=sizing_dict,
            geometry_summary=geo_dict.get("params", {}),
            surrogate_prediction=surrogate_dict,
        )
        version_id = save_version(version)
        pipeline_result["version_id"] = version_id

        # ── Done ──────────────────────────────────────────────────────────
        elapsed = round(time.perf_counter() - t0, 3)
        pipeline_result.update({
            "status": "completed",
            "elapsed_s": elapsed,
            # Convenience top-level fields
            "D2_mm": round(sizing_dict.get("impeller_d2", 0) * 1000, 1),
            "eta": sizing_dict.get("estimated_efficiency"),
            "eta_surrogate": surrogate_dict.get("eta_total"),
        })

        tracker.update_progress(run_id, 100, "done", "Pipeline completed")
        log.info("run_full_pipeline_task: done elapsed=%.3fs version_id=%s", elapsed, version_id)
        return pipeline_result

    except Exception as exc:
        log.exception("run_full_pipeline_task: failed — %s", exc)
        tracker.update_progress(run_id, 0, "failed", str(exc))
        pipeline_result.update({"status": "failed", "error": str(exc)})
        raise RuntimeError(f"Full pipeline failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Backwards-compatible legacy tasks (kept to avoid breaking existing workers)
# ---------------------------------------------------------------------------

@_task(bind=True, name="hpe.sizing")
def _legacy_run_sizing_task(self, flow_rate: float, head: float, rpm: float) -> dict:
    """Legacy sizing task — kept for backwards compatibility."""
    return run_sizing_task({"flow_rate": flow_rate, "head": head, "rpm": rpm})


@_task(bind=True, name="hpe.curves")
def _legacy_run_curves_task(self, flow_rate: float, head: float, rpm: float, n_points: int = 25) -> dict:
    """Legacy curves task — kept for backwards compatibility."""
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.physics.curves import generate_curves

    if _CELERY_AVAILABLE and hasattr(self, "update_state"):
        self.update_state(state="RUNNING", meta={"step": "curves"})

    op = OperatingPoint(flow_rate=flow_rate, head=head, rpm=rpm)
    sizing = run_sizing(op)
    curves = generate_curves(sizing, n_points=n_points)
    return {
        "flow_rates": curves.flow_rates,
        "heads": curves.heads,
        "efficiencies": curves.efficiencies,
        "powers": curves.powers,
    }


@_task(bind=True, name="hpe.optimize", time_limit=7200)
def _legacy_run_optimization_task(
    self, flow_rate: float, head: float, rpm: float,
    method: str = "nsga2", pop_size: int = 40, n_gen: int = 50,
) -> dict:
    """Legacy optimisation task — kept for backwards compatibility."""
    return run_optimization_task(
        {"flow_rate": flow_rate, "head": head, "rpm": rpm},
        method=method, pop_size=pop_size, n_gen=n_gen,
    )


@_task(bind=True, name="hpe.pipeline")
def _legacy_run_full_pipeline_task(self, flow_rate: float, head: float, rpm: float) -> dict:
    """Legacy full-pipeline task — kept for backwards compatibility."""
    return run_full_pipeline_task({"flow_rate": flow_rate, "head": head, "rpm": rpm})
