"""Tests for blade profile generation."""


import pytest

from hpe.geometry.models import RunnerGeometryParams
from hpe.geometry.runner.blade import (
    calc_wrap_angle,
    generate_blade_profile,
)


class TestBladeProfile:
    def test_generates_points(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        assert len(profile.camber_points) > 10
        assert len(profile.pressure_side) == len(profile.camber_points)
        assert len(profile.suction_side) == len(profile.camber_points)

    def test_inlet_radius(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        r_inlet = profile.camber_points[0][0]
        assert r_inlet == pytest.approx(runner_params.d1 / 2, rel=0.01)

    def test_outlet_radius(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        r_outlet = profile.camber_points[-1][0]
        assert r_outlet == pytest.approx(runner_params.d2 / 2, rel=0.01)

    def test_radius_increasing(self, runner_params: RunnerGeometryParams) -> None:
        """Radius should increase from inlet to outlet."""
        profile = generate_blade_profile(runner_params)
        for i in range(1, len(profile.camber_points)):
            assert profile.camber_points[i][0] >= profile.camber_points[i - 1][0] - 1e-10

    def test_theta_starts_at_zero(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        assert profile.camber_points[0][1] == 0.0

    def test_thickness_applied(self, runner_params: RunnerGeometryParams) -> None:
        """Pressure and suction sides should be offset from camber."""
        profile = generate_blade_profile(runner_params)
        # At midchord, offset should be maximum
        mid = len(profile.camber_points) // 2
        r_c, theta_c = profile.camber_points[mid]
        _, theta_p = profile.pressure_side[mid]
        _, theta_s = profile.suction_side[mid]

        # Pressure side should be at higher theta, suction at lower
        assert theta_p > theta_c
        assert theta_s < theta_c

    def test_zero_thickness_at_edges(self, runner_params: RunnerGeometryParams) -> None:
        """Thickness should be zero at leading and trailing edges."""
        profile = generate_blade_profile(runner_params)

        # Leading edge (index 0)
        _, theta_c_le = profile.camber_points[0]
        _, theta_p_le = profile.pressure_side[0]
        assert theta_p_le == pytest.approx(theta_c_le, abs=1e-10)

        # Trailing edge (last index)
        _, theta_c_te = profile.camber_points[-1]
        _, theta_p_te = profile.pressure_side[-1]
        assert theta_p_te == pytest.approx(theta_c_te, abs=1e-10)

    def test_thickness_stored(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        assert profile.thickness == runner_params.blade_thickness


class TestWrapAngle:
    def test_positive(self, runner_params: RunnerGeometryParams) -> None:
        profile = generate_blade_profile(runner_params)
        wrap = calc_wrap_angle(profile)
        assert wrap > 0

    def test_reasonable_range(self, runner_params: RunnerGeometryParams) -> None:
        """Wrap angle for centrifugal pump blades: typically 30-150 degrees."""
        profile = generate_blade_profile(runner_params)
        wrap = calc_wrap_angle(profile)
        assert 10 < wrap < 200
