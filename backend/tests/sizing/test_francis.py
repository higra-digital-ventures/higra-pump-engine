"""Tests for Francis turbine sizing."""


from hpe.sizing.francis import FrancisSizing, size_francis


class TestFrancisSizing:
    def test_basic_sizing(self) -> None:
        """Standard Francis: Q=2 m3/s, H=100m, 600 rpm."""
        result = size_francis(2.0, 100.0, 600)
        assert isinstance(result, FrancisSizing)

    def test_dimensions_positive(self) -> None:
        result = size_francis(2.0, 100.0, 600)
        assert result.d1 > 0
        assert result.d2 > 0
        assert result.b0 > 0
        assert result.d_draft > 0

    def test_d2_less_than_d1(self) -> None:
        """For Francis, D2 < D1 (flow goes inward)."""
        result = size_francis(2.0, 100.0, 600)
        assert result.d2 < result.d1

    def test_efficiency_range(self) -> None:
        result = size_francis(2.0, 100.0, 600)
        assert 0.80 < result.estimated_efficiency < 0.96

    def test_power_positive(self) -> None:
        result = size_francis(2.0, 100.0, 600)
        assert result.estimated_power > 0
        # P ~ rho * g * Q * H * eta ~ 998 * 9.81 * 2 * 100 * 0.93 ~ 1.8 MW
        assert result.estimated_power > 1e6

    def test_blade_count_range(self) -> None:
        result = size_francis(2.0, 100.0, 600)
        assert 9 <= result.blade_count <= 19

    def test_angles_reasonable(self) -> None:
        result = size_francis(2.0, 100.0, 600)
        assert 5 < result.alpha1 < 45
        # Francis beta1 can be >90 deg (backward-curved blades at inlet)
        assert 5 < result.beta1 < 170
        assert 5 < result.beta2 < 45

    def test_small_turbine(self) -> None:
        """Small Francis: Q=0.5 m3/s, H=50m."""
        result = size_francis(0.5, 50.0, 1000)
        assert result.d1 < 1.0  # Less than 1m diameter
        assert result.estimated_power > 0

    def test_large_turbine(self) -> None:
        """Large Francis: Q=50 m3/s, H=200m."""
        result = size_francis(50.0, 200.0, 300)
        assert result.d1 > 1.0  # Large runner
        assert result.estimated_power > 50e6  # >50 MW
