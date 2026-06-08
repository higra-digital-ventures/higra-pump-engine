"""Tests for Bayesian optimizer."""

import pytest

pytest.importorskip("optuna", reason="Bayesian optimizer requires Optuna")

from hpe.optimization.bayesian import run_bayesian
from hpe.optimization.problem import OptimizationProblem


class TestBayesian:
    def test_finds_best(self, default_problem: OptimizationProblem) -> None:
        result = run_bayesian(default_problem, n_trials=20, seed=42)
        assert "best_params" in result
        assert "best_value" in result
        assert result["best_value"] > 0

    def test_returns_params(self, default_problem: OptimizationProblem) -> None:
        result = run_bayesian(default_problem, n_trials=10, seed=42)
        assert "beta2" in result["best_params"]
        assert "blade_count" in result["best_params"]
