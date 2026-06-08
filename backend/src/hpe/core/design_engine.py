"""Constraint-driven design facade — the "computational engineering model" entry point.

This is the LEAP 71 / Noyron-style top-level function: given the *intent*
(operating point + engineering constraints), it produces a complete design that
is **expected to be feasible**, iterating automatically when a constraint is
violated — instead of leaving the engineer to tweak parameters by hand.

It is a thin orchestrator on top of two existing pieces:

* :class:`hpe.core.design_state.DesignState` — carries intent + results and
  knows how to evaluate feasibility (``evaluate_constraints``);
* the 1D sizing (``hpe.sizing.run_sizing``) — the physics generator.

Strategy
--------
Each iteration runs sizing, evaluates constraints, and — if infeasible —
inspects *which* constraints failed and adjusts the actionable knobs:

* **rotational speed** (the primary knob): lower it when NPSHr is too high or
  Nq too high; raise it when Nq is too low. Speed steps through standard
  motor speeds.
* **number of stages**: when the per-stage tip speed (u₂, set mainly by head)
  exceeds the material limit, split the head across more stages.

Visited ``(rpm, n_stages)`` pairs are remembered to avoid cycles; the best
attempt (feasible first, then highest efficiency / fewest violations) is kept.

The sizing call is injectable (``sizing_fn``) so the control logic is unit
testable without the heavy physics stack.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from hpe.core.design_state import DesignState, DesignConstraints

log = logging.getLogger(__name__)

# Standard induction-motor speeds [rpm] (50/60 Hz, 2–8 pole), high → low.
DEFAULT_RPM_CHOICES = (3500, 2900, 1750, 1450, 1160, 970, 880, 720)

SizingFn = Callable[[dict], dict]


@dataclass
class DesignAttempt:
    """A single iteration of the design loop."""

    iteration: int
    rpm: float
    n_stages: int
    feasible: bool
    failed: list[str]
    eta: Optional[float]
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DesignOutcome:
    """Result of a constraint-driven design run."""

    state: DesignState
    feasible: bool
    iterations: int
    attempts: list[DesignAttempt] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "iterations": self.iterations,
            "message": self.message,
            "summary": self.state.summary(),
            "attempts": [a.to_dict() for a in self.attempts],
            "state": self.state.to_dict(),
        }


def design(
    op: Any,
    constraints: Optional[DesignConstraints] = None,
    *,
    max_iterations: int = 8,
    allow_speed_adjust: bool = True,
    allow_multistage: bool = True,
    max_stages: int = 4,
    rpm_choices: Optional[tuple[int, ...]] = None,
    sizing_fn: Optional[SizingFn] = None,
) -> DesignOutcome:
    """Generate a feasible design for the given duty + constraints.

    Parameters
    ----------
    op : OperatingPoint | dict
        Required duty: ``flow_rate`` [m³/s], ``head`` [m], ``rpm`` (starting speed).
    constraints : DesignConstraints, optional
        Engineering limits (defaults applied if omitted).
    max_iterations : int
        Maximum number of sizing attempts.
    allow_speed_adjust, allow_multistage : bool
        Enable/disable each adjustment strategy.
    max_stages : int
        Upper bound on stage count when splitting head.
    rpm_choices : tuple[int], optional
        Discrete speeds to step through (default :data:`DEFAULT_RPM_CHOICES`).
    sizing_fn : callable, optional
        ``(op_dict) -> sizing_dict``. Defaults to the real 1D sizing.

    Returns
    -------
    DesignOutcome
        Best design found, its feasibility, and the full attempt trail.
    """
    constraints = constraints or DesignConstraints()
    rpm_choices = tuple(sorted(rpm_choices or DEFAULT_RPM_CHOICES, reverse=True))
    sizing_fn = sizing_fn or _default_sizing_fn

    base = DesignState.from_operating_point(op, constraints).operating_point
    total_head = float(base.get("head", 0.0))
    rpm = float(base.get("rpm", 0.0))
    n_stages = 1

    visited: set[tuple[float, int]] = set()
    attempts: list[DesignAttempt] = []
    best: Optional[tuple[DesignState, DesignAttempt]] = None

    for it in range(1, max_iterations + 1):
        key = (rpm, n_stages)
        if key in visited:
            log.debug("design: revisiting %s — stopping to avoid cycle", key)
            break
        visited.add(key)

        # Build the per-stage operating point (head splits across stages).
        op_dict = dict(base)
        op_dict["rpm"] = rpm
        op_dict["head"] = total_head / n_stages
        op_dict["n_stages"] = n_stages

        try:
            sizing = sizing_fn(op_dict)
        except Exception as exc:  # noqa: BLE001
            log.warning("design: sizing failed at rpm=%.0f n=%d (%s)", rpm, n_stages, exc)
            attempts.append(DesignAttempt(it, rpm, n_stages, False, ["sizing_error"], None, str(exc)))
            break

        state = DesignState.from_operating_point(op_dict, constraints)
        state.record_stage("sizing", sizing)
        report = state.evaluate_constraints()
        eta = _num(sizing.get("estimated_efficiency"))
        failed = [c.name for c in report.failed]

        attempt = DesignAttempt(
            iteration=it, rpm=rpm, n_stages=n_stages,
            feasible=report.feasible, failed=failed, eta=eta,
            note="; ".join(c.message for c in report.failed) or "ok",
        )
        attempts.append(attempt)

        if best is None or _is_better(attempt, best[1]):
            best = (state, attempt)

        if report.feasible:
            return DesignOutcome(
                state=state, feasible=True, iterations=it, attempts=attempts,
                message=f"Feasible design at rpm={rpm:.0f}, {n_stages} stage(s).",
            )

        # --- decide the next adjustment based on what failed ---
        nxt = _next_params(
            failed, rpm, n_stages, rpm_choices, max_stages,
            allow_speed_adjust, allow_multistage,
        )
        if nxt is None:
            break
        rpm, n_stages = nxt

    # No feasible design found — return the best attempt.
    if best is None:
        empty = DesignState.from_operating_point(op, constraints)
        return DesignOutcome(empty, False, len(attempts), attempts, "No design produced.")
    state, attempt = best
    return DesignOutcome(
        state=state, feasible=False, iterations=len(attempts), attempts=attempts,
        message=(
            f"No fully feasible design in {len(attempts)} attempt(s); "
            f"best at rpm={attempt.rpm:.0f}, {attempt.n_stages} stage(s) "
            f"(violations: {', '.join(attempt.failed) or 'none'})."
        ),
    )


# ---------------------------------------------------------------------------
# Adjustment policy
# ---------------------------------------------------------------------------

def _next_params(
    failed: list[str],
    rpm: float,
    n_stages: int,
    rpm_choices: tuple[int, ...],
    max_stages: int,
    allow_speed: bool,
    allow_multistage: bool,
) -> Optional[tuple[float, int]]:
    """Pick the next (rpm, n_stages) given the violated constraints.

    Priority order matters: tip speed (a head problem) is addressed by adding
    stages first; speed-related violations then nudge rpm.
    """
    fs = set(failed)

    # Tip speed too high → head per stage too high → add a stage.
    if "max_tip_speed" in fs and allow_multistage and n_stages < max_stages:
        return (rpm, n_stages + 1)

    if allow_speed:
        # Too fast (cavitation / Nq above range) → step down.
        if {"max_npsh_r", "nq_max"} & fs:
            lower = _step(rpm, rpm_choices, direction=-1)
            if lower is not None:
                return (lower, n_stages)
        # Too slow (Nq below range, or low efficiency from low Nq) → step up.
        if {"nq_min", "min_efficiency"} & fs:
            higher = _step(rpm, rpm_choices, direction=+1)
            if higher is not None:
                return (higher, n_stages)

    # Fallback: if tip speed still the issue but multistage exhausted, give up.
    return None


def _step(rpm: float, choices: tuple[int, ...], direction: int) -> Optional[float]:
    """Next discrete speed below (-1) or above (+1) the current rpm.

    ``choices`` is sorted high→low. Returns ``None`` at the boundary.
    """
    desc = list(choices)  # high → low
    # Find the nearest choice to the current rpm.
    nearest_idx = min(range(len(desc)), key=lambda i: abs(desc[i] - rpm))
    if direction < 0:        # lower speed = next index (smaller value)
        idx = nearest_idx + 1
    else:                    # higher speed = previous index (larger value)
        idx = nearest_idx - 1
    if 0 <= idx < len(desc) and desc[idx] != rpm:
        return float(desc[idx])
    return None


def _is_better(a: DesignAttempt, b: DesignAttempt) -> bool:
    """Ranking: feasible beats infeasible; then fewer violations; then higher η."""
    if a.feasible != b.feasible:
        return a.feasible
    if len(a.failed) != len(b.failed):
        return len(a.failed) < len(b.failed)
    return (a.eta or 0.0) > (b.eta or 0.0)


# ---------------------------------------------------------------------------
# Default sizing function (wraps the real 1D sizing)
# ---------------------------------------------------------------------------

def _default_sizing_fn(op_dict: dict) -> dict:
    """Run the real 1D meanline sizing and serialise the fields we need."""
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing

    op = OperatingPoint(
        flow_rate=float(op_dict["flow_rate"]),
        head=float(op_dict["head"]),
        rpm=float(op_dict["rpm"]),
    )
    r = run_sizing(op)
    return {
        "specific_speed_nq": getattr(r, "specific_speed_nq", None),
        "impeller_d2": getattr(r, "impeller_d2", None),
        "impeller_b2": getattr(r, "impeller_b2", None),
        "estimated_efficiency": getattr(r, "estimated_efficiency", None),
        "estimated_power": getattr(r, "estimated_power", None),
        "estimated_npsh_r": getattr(r, "estimated_npsh_r", None),
        "sigma": getattr(r, "sigma", None),
        "diffusion_ratio": getattr(r, "diffusion_ratio", 0.0),
        "blade_count": getattr(r, "blade_count", None),
        "velocity_triangles": getattr(r, "velocity_triangles", {}) or {},
    }


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
