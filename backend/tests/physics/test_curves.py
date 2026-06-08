"""Tests for performance curve generation."""


from hpe.core.models import SizingResult
from hpe.physics.curves import (
    PerformanceCurves,
    generate_curves,
    generate_efficiency_curve,
    generate_hq_curve,
)


class TestGenerateCurves:
    def test_returns_curves(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=10)
        assert isinstance(curves, PerformanceCurves)

    def test_correct_point_count(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=15)
        assert len(curves.flow_rates) == 15
        assert len(curves.heads) == 15
        assert len(curves.efficiencies) == 15
        assert len(curves.powers) == 15
        assert len(curves.npsh_required) == 15

    def test_flow_rates_increasing(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=10)
        for i in range(1, len(curves.flow_rates)):
            assert curves.flow_rates[i] > curves.flow_rates[i - 1]

    def test_heads_generally_decreasing(self, sizing_result: SizingResult) -> None:
        """H-Q curve should be mostly falling for centrifugal pump."""
        curves = generate_curves(sizing_result, n_points=15)
        # Check overall trend: first head > last head
        assert curves.heads[0] > curves.heads[-1]

    def test_all_heads_positive(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=15)
        for h in curves.heads:
            assert h >= 0

    def test_efficiency_has_peak(self, sizing_result: SizingResult) -> None:
        """Efficiency curve should have a peak (not monotonic)."""
        curves = generate_curves(sizing_result, n_points=20)
        max_idx = curves.efficiencies.index(max(curves.efficiencies))
        # Peak should not be at the first or last point
        assert 0 < max_idx < len(curves.efficiencies) - 1

    def test_metrics_populated(self, sizing_result: SizingResult) -> None:
        curves = generate_curves(sizing_result, n_points=5)
        assert len(curves.metrics) == 5
        assert curves.metrics[0].head > 0


class TestHQCurve:
    def test_returns_tuples(self, sizing_result: SizingResult) -> None:
        hq = generate_hq_curve(sizing_result, n_points=10)
        assert len(hq) == 10
        assert len(hq[0]) == 2  # (Q, H)

    def test_falling_trend(self, sizing_result: SizingResult) -> None:
        hq = generate_hq_curve(sizing_result, n_points=10)
        assert hq[0][1] > hq[-1][1]  # H at low Q > H at high Q


class TestEfficiencyCurve:
    def test_returns_tuples(self, sizing_result: SizingResult) -> None:
        eq = generate_efficiency_curve(sizing_result, n_points=10)
        assert len(eq) == 10
        assert 0 < eq[5][1] < 1  # Efficiency between 0 and 1
