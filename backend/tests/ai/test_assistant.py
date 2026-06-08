"""Tests for engineering assistant."""

import pytest

from hpe.ai.assistant.interpreter import interpret_performance, interpret_sizing
from hpe.ai.assistant.recommender import Recommendation, recommend_improvements
from hpe.core.models import OperatingPoint
from hpe.physics.euler import get_design_flow_rate
from hpe.physics.performance import evaluate_performance
from hpe.sizing import run_sizing


@pytest.fixture
def sizing():
    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
    return run_sizing(op)


class TestInterpreter:
    def test_interpret_sizing_returns_text(self, sizing) -> None:
        text = interpret_sizing(sizing)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_mentions_specific_speed(self, sizing) -> None:
        text = interpret_sizing(sizing)
        assert "Nq" in text or "specific speed" in text.lower()

    def test_mentions_efficiency(self, sizing) -> None:
        text = interpret_sizing(sizing)
        assert "efficiency" in text.lower() or "%" in text

    def test_mentions_dimensions(self, sizing) -> None:
        text = interpret_sizing(sizing)
        assert "mm" in text

    def test_interpret_performance(self, sizing) -> None:
        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        text = interpret_performance(perf, 30.0)
        assert isinstance(text, str)
        assert "efficiency" in text.lower()
        assert "head" in text.lower()

    def test_identifies_dominant_loss(self, sizing) -> None:
        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        text = interpret_performance(perf, 30.0)
        assert "dominant" in text.lower()


class TestRecommender:
    def test_returns_recommendations(self, sizing) -> None:
        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        recs = recommend_improvements(sizing, perf)
        assert isinstance(recs, list)

    def test_recommendations_have_structure(self, sizing) -> None:
        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        recs = recommend_improvements(sizing, perf)
        for rec in recs:
            assert isinstance(rec, Recommendation)
            assert rec.category in ["efficiency", "cavitation", "robustness", "manufacturing"]
            assert rec.priority in ["high", "medium", "low"]
            assert len(rec.reason) > 10

    def test_sorted_by_priority(self, sizing) -> None:
        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        recs = recommend_improvements(sizing, perf)
        if len(recs) >= 2:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(1, len(recs)):
                assert priority_order[recs[i].priority] >= priority_order[recs[i - 1].priority]

    def test_with_curves(self, sizing) -> None:
        from hpe.physics.curves import generate_curves

        q = get_design_flow_rate(sizing)
        perf = evaluate_performance(sizing, q)
        curves = generate_curves(sizing, n_points=10)
        recs = recommend_improvements(sizing, perf, curves)
        assert isinstance(recs, list)
