"""Tests for benchmark validation module."""

from __future__ import annotations


from hpe.validation.benchmark import (
    BenchmarkResult,
    TestBenchPoint,
    benchmark_sizing,
)


def _make_test_data() -> list[TestBenchPoint]:
    """Create synthetic test bench data for validation."""
    # Simulate a range of operating points with "measured" values
    # that are close to what HPE would predict (with some noise)
    points = []
    for q_m3h in [50, 100, 150, 200, 250]:
        q = q_m3h / 3600.0
        h = 30.0  # Fixed head
        rpm = 1750.0

        # Approximate expected values (will differ from HPE prediction)
        eta_approx = 0.70 + 0.05 * (q_m3h / 200)  # 70-76%
        p_approx = 998.2 * 9.81 * q * h / eta_approx

        points.append(TestBenchPoint(
            flow_rate=q,
            head=h,
            rpm=rpm,
            measured_efficiency=eta_approx,
            measured_power=p_approx,
            measured_npsh=3.0 + q_m3h * 0.01,
            machine_model="TEST-PUMP-001",
        ))
    return points


class TestBenchmark:
    def test_basic_run(self) -> None:
        data = _make_test_data()
        result = benchmark_sizing(data)
        assert isinstance(result, BenchmarkResult)
        assert result.n_points == 5

    def test_comparisons_populated(self) -> None:
        data = _make_test_data()
        result = benchmark_sizing(data)
        for c in result.comparisons:
            assert c.pred_efficiency > 0
            assert c.pred_power > 0
            assert c.pred_d2 > 0

    def test_efficiency_metrics(self) -> None:
        data = _make_test_data()
        result = benchmark_sizing(data)
        assert result.efficiency_metrics is not None
        assert result.efficiency_metrics.count == 5
        assert result.efficiency_metrics.mae >= 0
        assert result.efficiency_metrics.rmse >= 0
        # R² can be negative when model is worse than mean predictor
        assert result.efficiency_metrics.r_squared <= 1.0

    def test_power_metrics(self) -> None:
        data = _make_test_data()
        result = benchmark_sizing(data)
        assert result.power_metrics is not None
        assert result.power_metrics.count == 5

    def test_skip_errors(self) -> None:
        # Invalid point (zero flow) should be skipped
        data = [
            TestBenchPoint(flow_rate=0.0001, head=0.1, rpm=100,
                          measured_efficiency=0.5, measured_power=100),
            TestBenchPoint(flow_rate=0.05, head=30.0, rpm=1750,
                          measured_efficiency=0.75, measured_power=20000),
        ]
        result = benchmark_sizing(data, skip_errors=True)
        # At least one should succeed
        assert result.n_points >= 1

    def test_partial_measurements(self) -> None:
        data = [
            TestBenchPoint(flow_rate=0.05, head=30.0, rpm=1750,
                          measured_efficiency=0.75),  # No power or NPSH
        ]
        result = benchmark_sizing(data)
        assert result.n_points == 1
        assert result.comparisons[0].efficiency_error is not None
        assert result.comparisons[0].power_error_pct is None

    def test_within_thresholds(self) -> None:
        data = _make_test_data()
        result = benchmark_sizing(data)
        if result.efficiency_metrics:
            assert 0 <= result.efficiency_metrics.within_5pct <= 1
            assert 0 <= result.efficiency_metrics.within_10pct <= 1
            # 10% threshold should capture >= 5% threshold
            assert result.efficiency_metrics.within_10pct >= result.efficiency_metrics.within_5pct

    def test_empty_data(self) -> None:
        result = benchmark_sizing([])
        assert result.n_points == 0
        assert result.efficiency_metrics is None
