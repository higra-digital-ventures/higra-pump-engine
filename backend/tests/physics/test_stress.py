"""Tests for stress prediction module."""

from __future__ import annotations


import pytest

from hpe.core.models import OperatingPoint
from hpe.physics.stress import (
    BRONZE,
    CAST_IRON,
    DUPLEX_2205,
    STAINLESS_316L,
    StressResult,
    analyze_stress,
    calc_bending_stress,
    calc_blade_natural_frequency,
    calc_centrifugal_stress,
)
from hpe.sizing.meanline import run_sizing


@pytest.fixture
def sizing_result():
    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
    return run_sizing(op)


class TestCentrifugalStress:
    def test_positive(self) -> None:
        root, tip = calc_centrifugal_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7, rpm=1750,
        )
        assert root > 0
        assert tip > 0

    def test_root_greater_than_tip(self) -> None:
        root, tip = calc_centrifugal_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7, rpm=1750,
        )
        assert root > tip

    def test_higher_rpm_more_stress(self) -> None:
        root_low, _ = calc_centrifugal_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7, rpm=1750,
        )
        root_high, _ = calc_centrifugal_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7, rpm=3500,
        )
        # Stress scales with ω² (4x for 2x rpm)
        assert root_high > root_low * 3.5

    def test_larger_diameter_more_stress(self) -> None:
        root_small, _ = calc_centrifugal_stress(
            d1=0.10, d2=0.20, b2=0.020,
            blade_thickness=0.004, blade_count=7, rpm=1750,
        )
        root_large, _ = calc_centrifugal_stress(
            d1=0.16, d2=0.40, b2=0.025,
            blade_thickness=0.004, blade_count=7, rpm=1750,
        )
        assert root_large > root_small


class TestBendingStress:
    def test_positive(self) -> None:
        le, te, max_b = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        assert le > 0
        assert te > 0
        assert max_b > 0

    def test_le_greater_than_te(self) -> None:
        le, te, _ = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=7,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        assert le > te  # LE has 1.2x factor

    def test_thicker_blade_less_stress(self) -> None:
        _, _, max_thin = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.003, blade_count=7,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        _, _, max_thick = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.006, blade_count=7,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        assert max_thick < max_thin

    def test_more_blades_less_stress(self) -> None:
        _, _, max_5 = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=5,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        _, _, max_9 = calc_bending_stress(
            d1=0.16, d2=0.32, b2=0.025,
            blade_thickness=0.004, blade_count=9,
            rpm=1750, head=30.0, flow_rate=0.05,
        )
        assert max_9 < max_5


class TestNaturalFrequency:
    def test_positive(self) -> None:
        f1 = calc_blade_natural_frequency(0.16, 0.32, 0.025, 0.004)
        assert f1 > 0

    def test_thicker_blade_higher_freq(self) -> None:
        f_thin = calc_blade_natural_frequency(0.16, 0.32, 0.025, 0.003)
        f_thick = calc_blade_natural_frequency(0.16, 0.32, 0.025, 0.006)
        assert f_thick > f_thin

    def test_taller_blade_lower_freq(self) -> None:
        f_short = calc_blade_natural_frequency(0.16, 0.32, 0.015, 0.004)
        f_tall = calc_blade_natural_frequency(0.16, 0.32, 0.040, 0.004)
        assert f_short > f_tall


class TestFullStressAnalysis:
    def test_basic_analysis(self, sizing_result) -> None:
        result = analyze_stress(
            sizing_result, rpm=1750, head=30.0, flow_rate=0.05,
        )
        assert isinstance(result, StressResult)
        assert result.von_mises_max > 0
        assert result.sf_yield > 0
        assert result.first_natural_freq > 0

    def test_safe_for_typical_pump(self, sizing_result) -> None:
        result = analyze_stress(
            sizing_result, rpm=1750, head=30.0, flow_rate=0.05,
            material=STAINLESS_316L,
        )
        assert result.is_safe
        assert result.sf_yield >= 1.5

    def test_weaker_material_lower_sf(self, sizing_result) -> None:
        result_ss = analyze_stress(
            sizing_result, rpm=1750, head=30.0, flow_rate=0.05,
            material=STAINLESS_316L,
        )
        result_ci = analyze_stress(
            sizing_result, rpm=1750, head=30.0, flow_rate=0.05,
            material=CAST_IRON,
        )
        assert result_ci.sf_yield < result_ss.sf_yield

    def test_high_rpm_generates_warnings(self, sizing_result) -> None:
        result = analyze_stress(
            sizing_result, rpm=5000, head=30.0, flow_rate=0.05,
            material=CAST_IRON,
        )
        assert len(result.warnings) > 0

    def test_material_presets_exist(self) -> None:
        for mat in [CAST_IRON, BRONZE, STAINLESS_316L, DUPLEX_2205]:
            assert mat.density > 0
            assert mat.yield_strength > 0
            assert mat.fatigue_limit > 0
