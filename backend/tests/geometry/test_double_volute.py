"""Tests for double volute and twin-entry volute."""

from __future__ import annotations


import pytest

from hpe.geometry.volute.double_volute import (
    DoubleVoluteParams,
    DoubleVoluteResult,
    VoluteType,
    calc_radial_force_double,
    calc_radial_force_single,
    size_double_volute,
)
from hpe.geometry.volute.models import CrossSectionType, VoluteParams


@pytest.fixture
def base_volute_params() -> VoluteParams:
    return VoluteParams(
        d2=0.32, b2=0.025,
        flow_rate=0.05, cu2=12.0,
    )


@pytest.fixture
def double_params(base_volute_params: VoluteParams) -> DoubleVoluteParams:
    return DoubleVoluteParams(
        base=base_volute_params,
        volute_type=VoluteType.DOUBLE,
    )


@pytest.fixture
def twin_params(base_volute_params: VoluteParams) -> DoubleVoluteParams:
    return DoubleVoluteParams(
        base=base_volute_params,
        volute_type=VoluteType.TWIN_ENTRY,
    )


class TestDoubleVolute:
    def test_basic_sizing(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        assert isinstance(result, DoubleVoluteResult)
        assert len(result.passage_a.areas) > 0
        assert len(result.passage_b.areas) > 0

    def test_two_passages(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        # Passage A: 0 → 180°
        assert result.passage_a.theta_stations[0] == pytest.approx(0.0)
        assert result.passage_a.theta_stations[-1] == pytest.approx(180.0)
        # Passage B: 180 → 360°
        assert result.passage_b.theta_stations[0] == pytest.approx(180.0)
        assert result.passage_b.theta_stations[-1] == pytest.approx(360.0)

    def test_areas_grow_monotonically(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        for passage in [result.passage_a, result.passage_b]:
            for i in range(1, len(passage.areas)):
                assert passage.areas[i] >= passage.areas[i - 1] - 1e-10

    def test_discharge_areas_similar(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        # Both passages carry 50% flow, so discharge areas should be similar
        ratio = result.merge_inlet_area_a / result.merge_inlet_area_b
        assert 0.8 < ratio < 1.2

    def test_radial_force_reduced(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        assert result.radial_force_ratio < 0.5  # At least 50% reduction

    def test_merge_area_larger(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        sum_inlets = result.merge_inlet_area_a + result.merge_inlet_area_b
        assert result.merge_outlet_area > sum_inlets  # Diffusion in merge

    def test_splitter_geometry(self, double_params: DoubleVoluteParams) -> None:
        result = size_double_volute(double_params)
        assert result.splitter_r_inner < result.splitter_r_outer
        assert result.splitter_r_inner > double_params.base.r3


class TestTwinEntry:
    def test_basic_sizing(self, twin_params: DoubleVoluteParams) -> None:
        result = size_double_volute(twin_params)
        assert isinstance(result, DoubleVoluteResult)

    def test_both_passages_full_360(self, twin_params: DoubleVoluteParams) -> None:
        result = size_double_volute(twin_params)
        # Twin entry: both passages span 0-360°
        assert result.passage_a.theta_stations[-1] == pytest.approx(360.0)
        assert result.passage_b.theta_stations[-1] == pytest.approx(360.0)

    def test_uneven_split(self, base_volute_params: VoluteParams) -> None:
        params = DoubleVoluteParams(
            base=base_volute_params,
            volute_type=VoluteType.TWIN_ENTRY,
            entry_flow_split=0.60,  # 60/40 split
        )
        result = size_double_volute(params)
        # Entry A should be larger than B
        assert result.merge_inlet_area_a > result.merge_inlet_area_b


class TestRadialForce:
    def test_single_positive(self, base_volute_params: VoluteParams) -> None:
        f = calc_radial_force_single(base_volute_params, q_actual=0.03)
        assert f > 0

    def test_single_zero_at_design(self, base_volute_params: VoluteParams) -> None:
        f = calc_radial_force_single(base_volute_params, q_actual=0.05)
        assert f < 1.0  # Should be very small at design point

    def test_double_less_than_single(self, double_params: DoubleVoluteParams) -> None:
        f_single = calc_radial_force_single(double_params.base, q_actual=0.03)
        f_double = calc_radial_force_double(double_params, q_actual=0.03)
        assert f_double < f_single

    def test_splitter_at_180_best(self) -> None:
        base = VoluteParams(d2=0.32, b2=0.025, flow_rate=0.05, cu2=12.0)
        p180 = DoubleVoluteParams(base=base, splitter_angle=180.0)
        p120 = DoubleVoluteParams(base=base, splitter_angle=120.0)

        r180 = size_double_volute(p180)
        r120 = size_double_volute(p120)

        assert r180.radial_force_ratio < r120.radial_force_ratio


class TestCrossSections:
    def test_trapezoidal_sections(self, base_volute_params: VoluteParams) -> None:
        base_volute_params.cross_section = CrossSectionType.TRAPEZOIDAL
        params = DoubleVoluteParams(base=base_volute_params)
        result = size_double_volute(params)
        assert len(result.passage_a.areas) > 0
        assert all(w >= 0 for w in result.passage_a.widths)

    def test_rectangular_sections(self, base_volute_params: VoluteParams) -> None:
        base_volute_params.cross_section = CrossSectionType.RECTANGULAR
        params = DoubleVoluteParams(base=base_volute_params)
        result = size_double_volute(params)
        assert len(result.passage_a.areas) > 0
