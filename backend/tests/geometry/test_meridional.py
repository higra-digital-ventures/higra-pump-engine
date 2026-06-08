"""Tests for meridional channel generation."""


import pytest

from hpe.geometry.models import RunnerGeometryParams
from hpe.geometry.runner.meridional import (
    calc_channel_width,
    generate_meridional_channel,
)


class TestMeridionalChannel:
    def test_generates_points(self, runner_params: RunnerGeometryParams) -> None:
        channel = generate_meridional_channel(runner_params)
        assert len(channel.hub_points) > 10
        assert len(channel.shroud_points) > 10

    def test_equal_point_count(self, runner_params: RunnerGeometryParams) -> None:
        channel = generate_meridional_channel(runner_params)
        assert len(channel.hub_points) == len(channel.shroud_points)

    def test_inlet_radius_correct(self, runner_params: RunnerGeometryParams) -> None:
        """First point should be at inlet radius."""
        channel = generate_meridional_channel(runner_params)
        r_hub_inlet = channel.hub_points[0][0]
        r_shroud_inlet = channel.shroud_points[0][0]

        assert r_hub_inlet == pytest.approx(runner_params.d1_hub / 2, rel=0.01)
        assert r_shroud_inlet == pytest.approx(runner_params.d1 / 2, rel=0.01)

    def test_outlet_radius_correct(self, runner_params: RunnerGeometryParams) -> None:
        """Last point should be at outlet radius."""
        channel = generate_meridional_channel(runner_params)
        r_hub_outlet = channel.hub_points[-1][0]
        r_shroud_outlet = channel.shroud_points[-1][0]

        assert r_hub_outlet == pytest.approx(runner_params.d2 / 2, rel=0.01)
        assert r_shroud_outlet == pytest.approx(runner_params.d2 / 2, rel=0.01)

    def test_radius_monotonically_increasing(self, runner_params: RunnerGeometryParams) -> None:
        """Radial coordinate should increase from inlet to outlet."""
        channel = generate_meridional_channel(runner_params)
        for points in [channel.hub_points, channel.shroud_points]:
            for i in range(1, len(points)):
                assert points[i][0] >= points[i - 1][0] - 1e-10

    def test_channel_width_positive(self, runner_params: RunnerGeometryParams) -> None:
        channel = generate_meridional_channel(runner_params)
        for i in range(len(channel.hub_points)):
            w = calc_channel_width(channel, i)
            assert w > 0

    def test_custom_n_points(self, runner_params: RunnerGeometryParams) -> None:
        channel = generate_meridional_channel(runner_params, n_points=10)
        assert len(channel.hub_points) == 10
