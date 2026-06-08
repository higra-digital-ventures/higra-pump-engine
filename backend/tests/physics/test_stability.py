"""Tests for stability analysis."""


from hpe.core.models import SizingResult
from hpe.physics.curves import generate_curves
from hpe.physics.euler import get_design_flow_rate
from hpe.physics.stability import (
    StabilityAnalysis,
    analyze_stability,
    find_bep,
)


class TestAnalyzeStability:
    def test_returns_analysis(self, sizing_result: SizingResult) -> None:
        analysis = analyze_stability(sizing_result)
        assert isinstance(analysis, StabilityAnalysis)

    def test_bep_positive(self, sizing_result: SizingResult) -> None:
        analysis = analyze_stability(sizing_result)
        assert analysis.bep_flow > 0
        assert analysis.bep_head > 0
        assert 0 < analysis.bep_efficiency < 1

    def test_bep_near_design(self, sizing_result: SizingResult) -> None:
        """BEP should be near the design point."""
        q_design = get_design_flow_rate(sizing_result)
        analysis = analyze_stability(sizing_result)
        ratio = analysis.bep_flow / q_design
        assert 0.5 < ratio < 1.5

    def test_shutdown_head_above_design(self, sizing_result: SizingResult) -> None:
        """Shutdown head should be above design head."""
        analysis = analyze_stability(sizing_result)
        assert analysis.shutdown_head > 0

    def test_min_flow_positive(self, sizing_result: SizingResult) -> None:
        analysis = analyze_stability(sizing_result)
        assert analysis.min_flow > 0
        assert 0 < analysis.min_flow_ratio < 1.0

    def test_warnings_are_list(self, sizing_result: SizingResult) -> None:
        analysis = analyze_stability(sizing_result)
        assert isinstance(analysis.warnings, list)


class TestFindBEP:
    def test_returns_tuple(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=15)
        q_bep, h_bep, eta_max = find_bep(curves)
        assert q_bep > 0
        assert h_bep > 0
        assert 0 < eta_max < 1

    def test_eta_max_is_maximum(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=15)
        _, _, eta_max = find_bep(curves)
        assert eta_max == max(curves.efficiencies)
