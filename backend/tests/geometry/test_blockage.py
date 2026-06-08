"""Tests for blockage model."""

from __future__ import annotations

import pytest

from hpe.geometry.runner.blockage import (
    BlockageMethod,
    BlockageSpec,
    apply_blockage_to_velocity,
    compute_blockage,
)


class TestConstantBlockage:
    def test_uniform_value(self) -> None:
        spec = BlockageSpec(method=BlockageMethod.CONSTANT, constant_value=0.88)
        result = compute_blockage(spec, n_points=10)
        assert all(abs(b - 0.88) < 1e-10 for b in result.blockage_factors)

    def test_mean_equals_constant(self) -> None:
        spec = BlockageSpec(method=BlockageMethod.CONSTANT, constant_value=0.92)
        result = compute_blockage(spec, n_points=20)
        assert result.mean_blockage == pytest.approx(0.92)


class TestTableBlockage:
    def test_interpolation(self) -> None:
        spec = BlockageSpec(
            method=BlockageMethod.TABLE,
            table_points=[(0.0, 0.95), (0.5, 0.88), (1.0, 0.85)],
        )
        result = compute_blockage(spec, m_coords=[0.0, 0.25, 0.5, 0.75, 1.0])
        assert abs(result.blockage_factors[0] - 0.95) < 1e-10
        assert abs(result.blockage_factors[2] - 0.88) < 1e-10
        assert abs(result.blockage_factors[4] - 0.85) < 1e-10

    def test_monotonic_decrease(self) -> None:
        spec = BlockageSpec(
            method=BlockageMethod.TABLE,
            table_points=[(0.0, 0.95), (1.0, 0.80)],
        )
        result = compute_blockage(spec, n_points=10)
        for i in range(1, len(result.blockage_factors)):
            assert result.blockage_factors[i] <= result.blockage_factors[i - 1] + 1e-10

    def test_empty_table_fallback(self) -> None:
        spec = BlockageSpec(method=BlockageMethod.TABLE, table_points=[])
        result = compute_blockage(spec, n_points=5)
        assert all(abs(b - 0.90) < 1e-10 for b in result.blockage_factors)


class TestCorrelationBlockage:
    def test_physical_range(self) -> None:
        spec = BlockageSpec(method=BlockageMethod.CORRELATION)
        result = compute_blockage(
            spec, n_points=20, d1=0.16, d2=0.32, b2=0.025, rpm=1750,
        )
        assert all(0.60 <= b <= 1.0 for b in result.blockage_factors)

    def test_decreases_along_passage(self) -> None:
        spec = BlockageSpec(method=BlockageMethod.CORRELATION)
        result = compute_blockage(
            spec, n_points=20, d1=0.16, d2=0.32, b2=0.025, rpm=1750,
        )
        # Blockage generally decreases (more blocked) toward outlet
        assert result.blockage_factors[0] >= result.blockage_factors[-1]

    def test_more_blades_more_blockage(self) -> None:
        spec_5 = BlockageSpec(method=BlockageMethod.CORRELATION, blade_count=5)
        spec_11 = BlockageSpec(method=BlockageMethod.CORRELATION, blade_count=11)
        r5 = compute_blockage(spec_5, n_points=10, d1=0.16, d2=0.32, b2=0.025)
        r11 = compute_blockage(spec_11, n_points=10, d1=0.16, d2=0.32, b2=0.025)
        assert r11.mean_blockage < r5.mean_blockage  # More blades = more blocked

    def test_thicker_blade_more_blockage(self) -> None:
        spec_thin = BlockageSpec(method=BlockageMethod.CORRELATION, blade_thickness=0.002)
        spec_thick = BlockageSpec(method=BlockageMethod.CORRELATION, blade_thickness=0.008)
        r_thin = compute_blockage(spec_thin, n_points=10, d1=0.16, d2=0.32, b2=0.025)
        r_thick = compute_blockage(spec_thick, n_points=10, d1=0.16, d2=0.32, b2=0.025)
        assert r_thick.mean_blockage < r_thin.mean_blockage


class TestApplyBlockage:
    def test_increases_velocity(self) -> None:
        cm = [5.0, 5.0, 5.0]
        spec = BlockageSpec(method=BlockageMethod.CONSTANT, constant_value=0.90)
        blockage = compute_blockage(spec, m_coords=[0.0, 0.5, 1.0])
        corrected = apply_blockage_to_velocity(cm, blockage)
        assert all(c > v for c, v in zip(corrected, cm))

    def test_correction_factor(self) -> None:
        cm = [10.0]
        spec = BlockageSpec(method=BlockageMethod.CONSTANT, constant_value=0.80)
        blockage = compute_blockage(spec, m_coords=[0.5])
        corrected = apply_blockage_to_velocity(cm, blockage)
        assert corrected[0] == pytest.approx(12.5)  # 10 / 0.8
