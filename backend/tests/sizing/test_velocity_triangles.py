"""Tests for velocity triangle calculations."""

import math

import pytest

from hpe.sizing.velocity_triangles import (
    calc_euler_head,
    calc_inlet_triangle,
    calc_outlet_triangle,
    calc_peripheral_velocity,
    calc_wiesner_slip_factor,
)


class TestPeripheralVelocity:
    def test_known_value(self) -> None:
        """D=0.3m, 1750rpm => u = pi*0.3*1750/60 ~ 27.49 m/s."""
        u = calc_peripheral_velocity(0.3, 1750)
        assert u == pytest.approx(math.pi * 0.3 * 1750 / 60, rel=1e-10)

    def test_zero_diameter(self) -> None:
        assert calc_peripheral_velocity(0.0, 1750) == 0.0


class TestInletTriangle:
    def test_no_preswirl(self) -> None:
        """Without pre-swirl, alpha1 = 90 deg and cu1 = 0."""
        tri = calc_inlet_triangle(0.1, 0.03, 0.05, 1750)
        assert tri.cu == 0.0
        assert tri.alpha == 90.0
        assert tri.cm > 0

    def test_velocity_consistency(self) -> None:
        """c^2 = cm^2 + cu^2 and w^2 = cm^2 + wu^2."""
        tri = calc_inlet_triangle(0.12, 0.035, 0.05, 1750)
        assert tri.c == pytest.approx(math.sqrt(tri.cm**2 + tri.cu**2), rel=1e-6)
        assert tri.w == pytest.approx(math.sqrt(tri.cm**2 + tri.wu**2), rel=1e-6)

    def test_wu_relation(self) -> None:
        """wu1 = u1 - cu1."""
        tri = calc_inlet_triangle(0.12, 0.035, 0.05, 1750)
        assert tri.wu == pytest.approx(tri.u - tri.cu, rel=1e-10)


class TestOutletTriangle:
    def test_cu2_positive(self) -> None:
        """For a pump, cu2 should be positive (energy added)."""
        tri = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0)
        assert tri.cu > 0

    def test_velocity_consistency(self) -> None:
        tri = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0)
        assert tri.c == pytest.approx(math.sqrt(tri.cm**2 + tri.cu**2), rel=1e-6)
        assert tri.w == pytest.approx(math.sqrt(tri.cm**2 + tri.wu**2), rel=1e-6)

    def test_slip_reduces_cu2(self) -> None:
        """Slip factor < 1 means actual cu2 < blade-congruent cu2."""
        tri_no_slip = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0, slip_factor=1.0)
        tri_with_slip = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0, slip_factor=0.8)
        assert tri_with_slip.cu < tri_no_slip.cu


class TestWiesnerSlipFactor:
    def test_range(self) -> None:
        """Slip factor should be between 0.5 and 0.95."""
        sigma = calc_wiesner_slip_factor(25.0, 7)
        assert 0.5 <= sigma <= 0.95

    def test_more_blades_higher_slip(self) -> None:
        """More blades => better flow guidance => higher slip factor."""
        sigma_5 = calc_wiesner_slip_factor(25.0, 5)
        sigma_9 = calc_wiesner_slip_factor(25.0, 9)
        assert sigma_9 > sigma_5


class TestEulerHead:
    def test_positive_for_pump(self) -> None:
        """Euler head should be positive for a pump (energy added)."""
        tri_in = calc_inlet_triangle(0.12, 0.035, 0.05, 1750)
        tri_out = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0)
        h_euler = calc_euler_head(tri_in, tri_out)
        assert h_euler > 0

    def test_reasonable_magnitude(self) -> None:
        """For Q=0.05, H~30m pump, Euler head should be ~30-45m."""
        tri_in = calc_inlet_triangle(0.12, 0.035, 0.05, 1750)
        tri_out = calc_outlet_triangle(0.25, 0.02, 0.05, 1750, 25.0)
        h_euler = calc_euler_head(tri_in, tri_out)
        assert 15 < h_euler < 80
