"""Integration tests for the meanline sizing orchestrator."""


from hpe.core.enums import MachineType
from hpe.core.models import OperatingPoint
from hpe.sizing.meanline import run_sizing


class TestRunSizing:
    """End-to-end tests for the run_sizing function."""

    def test_centrifugal_pump_basic(self, centrifugal_pump_op: OperatingPoint) -> None:
        """Basic centrifugal pump: Q=0.05 m3/s, H=30m, 1750 rpm."""
        result = run_sizing(centrifugal_pump_op)

        # Specific speed in centrifugal range
        assert 15 < result.specific_speed_nq < 60

        # Reasonable impeller diameter (200-350mm)
        assert 0.15 < result.impeller_d2 < 0.40

        # D1 < D2
        assert result.impeller_d1 < result.impeller_d2

        # Outlet width positive and reasonable
        assert 0.005 < result.impeller_b2 < 0.10

        # Blade count 5-12
        assert 5 <= result.blade_count <= 12

        # Blade angles in valid ranges
        assert 10 < result.beta1 < 60
        assert 15 < result.beta2 < 40

        # Efficiency 65-90%
        assert 0.65 < result.estimated_efficiency < 0.90

        # Power ~15-35 kW for this operating point
        power_kw = result.estimated_power / 1000
        assert 10 < power_kw < 50

        # NPSHr 1-15m
        assert 0.5 < result.estimated_npsh_r < 20

        # Sigma positive
        assert result.sigma > 0

    def test_velocity_triangles_populated(self, centrifugal_pump_op: OperatingPoint) -> None:
        result = run_sizing(centrifugal_pump_op)
        assert "inlet" in result.velocity_triangles
        assert "outlet" in result.velocity_triangles
        assert "euler_head" in result.velocity_triangles
        assert result.velocity_triangles["euler_head"] > 0

    def test_meridional_profile_populated(self, centrifugal_pump_op: OperatingPoint) -> None:
        result = run_sizing(centrifugal_pump_op)
        assert "d1" in result.meridional_profile
        assert "d2" in result.meridional_profile
        assert "b1" in result.meridional_profile
        assert "b2" in result.meridional_profile

    def test_small_pump(self) -> None:
        """Very small pump: Q=0.002 m3/s, H=15m, 2900 rpm."""
        op = OperatingPoint(flow_rate=0.002, head=15.0, rpm=2900)
        result = run_sizing(op)
        assert result.impeller_d2 < 0.20  # Small pump
        assert result.estimated_power > 0

    def test_large_pump(self) -> None:
        """Large pump: Q=0.5 m3/s, H=50m, 1450 rpm."""
        op = OperatingPoint(flow_rate=0.5, head=50.0, rpm=1450)
        result = run_sizing(op)
        assert result.impeller_d2 > 0.30  # Large pump
        assert result.estimated_efficiency > 0.70  # Better efficiency at scale

    def test_high_nq_generates_warning(self) -> None:
        """Very high Nq with centrifugal type should warn."""
        op = OperatingPoint(
            flow_rate=2.0, head=3.0, rpm=1450,
            machine_type=MachineType.CENTRIFUGAL_PUMP,
        )
        result = run_sizing(op)
        assert result.specific_speed_nq > 100
        # Should have a warning about machine type
        assert any("centrifugal" in w.lower() or "mixed" in w.lower() for w in result.warnings)

    def test_no_negative_dimensions(self, centrifugal_pump_op: OperatingPoint) -> None:
        """All dimensions must be positive."""
        result = run_sizing(centrifugal_pump_op)
        assert result.impeller_d2 > 0
        assert result.impeller_d1 > 0
        assert result.impeller_b2 > 0
        assert result.estimated_power > 0
        assert result.estimated_npsh_r > 0

    def test_euler_head_reasonable(self, centrifugal_pump_op: OperatingPoint) -> None:
        """Euler head should be somewhat above required head (accounts for losses)."""
        result = run_sizing(centrifugal_pump_op)
        h_euler = result.velocity_triangles["euler_head"]
        # Euler head should be > H (need to overcome losses)
        # and not more than ~2x H
        assert h_euler > centrifugal_pump_op.head * 0.8
        assert h_euler < centrifugal_pump_op.head * 2.5
