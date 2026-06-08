"""Tests for hpe.core.design_engine — the constraint-driven design facade.

A synthetic ``sizing_fn`` makes the control loop deterministic and lets us
assert *which* adjustment the engine chooses for each violated constraint,
without running the real physics stack.
"""

from __future__ import annotations

import math

import pytest

from hpe.core.design_engine import design, _step, DEFAULT_RPM_CHOICES
from hpe.core.design_state import DesignConstraints


def _sizing(*, nq=40.0, eta=0.82, npsh=3.0, sigma=0.1, u2=30.0, dr=0.78):
    """Build a sizing dict with explicit values."""
    return {
        "specific_speed_nq": nq,
        "impeller_d2": 0.30,
        "estimated_efficiency": eta,
        "estimated_npsh_r": npsh,
        "sigma": sigma,
        "diffusion_ratio": dr,
        "velocity_triangles": {"outlet": {"u": u2}},
    }


OP = {"flow_rate": 0.05, "head": 60.0, "rpm": 1750}


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_already_feasible_returns_first_iteration():
    out = design(OP, sizing_fn=lambda op: _sizing())
    assert out.feasible is True
    assert out.iterations == 1
    assert out.state.is_feasible


# ---------------------------------------------------------------------------
# NPSHr too high → step speed down
# ---------------------------------------------------------------------------

def test_high_npsh_steps_speed_down():
    # NPSHr scales with rpm; only ≤1750 rpm satisfies max_npsh_r=5.
    def sizing_fn(op):
        return _sizing(npsh=op["rpm"] / 500.0)  # 3500→7, 2900→5.8, 1750→3.5

    out = design(
        {**OP, "rpm": 3500},
        constraints=DesignConstraints(max_npsh_r=5.0),
        sizing_fn=sizing_fn,
    )
    assert out.feasible is True
    assert out.state.operating_point["rpm"] == 1750
    # Stepped 3500 → 2900 → 1750.
    assert [a.rpm for a in out.attempts] == [3500, 2900, 1750]


# ---------------------------------------------------------------------------
# tip speed too high → add stages
# ---------------------------------------------------------------------------

def test_high_tip_speed_adds_stages():
    # u2 set by per-stage head; splitting head lowers it below 45 m/s.
    def sizing_fn(op):
        u2 = 50.0 / math.sqrt(op["n_stages"])  # 1→50, 2→35.4
        return _sizing(u2=u2)

    out = design(OP, sizing_fn=sizing_fn)  # default max_tip_speed=45
    assert out.feasible is True
    assert out.state.operating_point["n_stages"] == 2
    assert [a.n_stages for a in out.attempts] == [1, 2]


# ---------------------------------------------------------------------------
# Nq below range → step speed up
# ---------------------------------------------------------------------------

def test_low_nq_steps_speed_up():
    def sizing_fn(op):
        return _sizing(nq=op["rpm"] / 50.0)  # 1450→29, 1750→35, 2900→58

    out = design(
        {**OP, "rpm": 1450},
        constraints=DesignConstraints(nq_min=40.0, nq_max=250.0),
        sizing_fn=sizing_fn,
    )
    assert out.feasible is True
    assert out.state.operating_point["rpm"] == 2900
    assert [a.rpm for a in out.attempts] == [1450, 1750, 2900]


# ---------------------------------------------------------------------------
# never feasible → returns best attempt, not an exception
# ---------------------------------------------------------------------------

def test_infeasible_returns_best_attempt():
    # Efficiency is hopeless regardless of speed.
    out = design(OP, sizing_fn=lambda op: _sizing(eta=0.50))
    assert out.feasible is False
    assert out.attempts                      # tried at least once
    assert "min_efficiency" in out.attempts[0].failed
    assert "No fully feasible design" in out.message


def test_max_iterations_respected():
    calls = {"n": 0}

    def sizing_fn(op):
        calls["n"] += 1
        return _sizing(eta=0.50)  # always fails efficiency

    out = design(OP, sizing_fn=sizing_fn, max_iterations=3)
    assert out.feasible is False
    assert calls["n"] <= 3


def test_sizing_error_is_captured_not_raised():
    def boom(op):
        raise RuntimeError("solver blew up")

    out = design(OP, sizing_fn=boom)
    assert out.feasible is False
    assert out.attempts[-1].failed == ["sizing_error"]


# ---------------------------------------------------------------------------
# outcome serialisation
# ---------------------------------------------------------------------------

def test_outcome_to_dict_shape():
    out = design(OP, sizing_fn=lambda op: _sizing())
    d = out.to_dict()
    assert d["feasible"] is True
    assert "summary" in d and "attempts" in d and "state" in d
    assert d["summary"]["operating_point"]["Q_m3h"] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# _step helper
# ---------------------------------------------------------------------------

def test_step_down_and_up():
    assert _step(3500, DEFAULT_RPM_CHOICES, -1) == 2900
    assert _step(1750, DEFAULT_RPM_CHOICES, +1) == 2900
    assert _step(3500, DEFAULT_RPM_CHOICES, +1) is None   # already fastest
    assert _step(720, DEFAULT_RPM_CHOICES, -1) is None    # already slowest
