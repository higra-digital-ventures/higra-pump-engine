"""Tests for blade stacking module."""

from __future__ import annotations

import math

import pytest

from hpe.geometry.runner.stacking import (
    StackingSpec,
    StackingType,
    apply_stacking_to_sections,
    compute_stacking,
)


class TestRadialStacking:
    def test_no_offset(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.RADIAL)
        result = compute_stacking(spec, n_spans=5)
        assert all(abs(o) < 1e-10 for o in result.theta_offsets)

    def test_correct_spans(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.RADIAL)
        result = compute_stacking(spec, n_spans=5)
        assert len(result.span_fractions) == 5
        assert result.span_fractions[0] == pytest.approx(0.0)
        assert result.span_fractions[-1] == pytest.approx(1.0)


class TestLeanStacking:
    def test_linear_distribution(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.LEAN, lean_angle=10.0)
        result = compute_stacking(spec, n_spans=5)
        assert abs(result.theta_offsets[0]) < 1e-10  # Hub = 0
        assert abs(result.theta_offsets[-1] - 10.0) < 1e-10  # Shroud = lean_angle

    def test_negative_lean(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.LEAN, lean_angle=-5.0)
        result = compute_stacking(spec, n_spans=3)
        assert result.theta_offsets[-1] < 0

    def test_max_lean_metric(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.LEAN, lean_angle=15.0)
        result = compute_stacking(spec, n_spans=5)
        assert result.max_lean_angle == pytest.approx(15.0)


class TestBowStacking:
    def test_zero_at_endpoints(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.BOW, bow_angle=8.0)
        result = compute_stacking(spec, n_spans=5)
        assert abs(result.theta_offsets[0]) < 1e-10
        assert abs(result.theta_offsets[-1]) < 1e-10

    def test_max_at_midspan(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.BOW, bow_angle=8.0)
        result = compute_stacking(spec, n_spans=5)
        mid = len(result.theta_offsets) // 2
        assert result.theta_offsets[mid] == pytest.approx(8.0)

    def test_bow_ratio(self) -> None:
        spec = StackingSpec(stacking_type=StackingType.BOW, bow_angle=8.0)
        result = compute_stacking(spec, n_spans=5)
        assert result.le_bow_ratio > 0


class TestCustomStacking:
    def test_interpolation(self) -> None:
        spec = StackingSpec(
            stacking_type=StackingType.CUSTOM,
            custom_points=[(0.0, 0.0), (0.5, 5.0), (1.0, 3.0)],
        )
        result = compute_stacking(spec, n_spans=5)
        assert abs(result.theta_offsets[0]) < 1e-10
        assert abs(result.theta_offsets[2] - 5.0) < 0.5  # midspan
        assert abs(result.theta_offsets[-1] - 3.0) < 1e-10


class TestApplyStacking:
    def test_applies_offsets(self) -> None:
        sections = [
            [(0.08, 0.0), (0.10, 0.1), (0.16, 0.3)],
            [(0.08, 0.0), (0.10, 0.1), (0.16, 0.3)],
            [(0.08, 0.0), (0.10, 0.1), (0.16, 0.3)],
        ]
        spec = StackingSpec(stacking_type=StackingType.LEAN, lean_angle=10.0)
        stacking = compute_stacking(spec, n_spans=3)

        stacked = apply_stacking_to_sections(sections, stacking)

        # Hub should be unchanged
        assert stacked[0][0][1] == pytest.approx(0.0)
        # Shroud should be offset
        assert stacked[-1][0][1] == pytest.approx(math.radians(10.0), rel=1e-3)

    def test_mismatch_raises(self) -> None:
        sections = [[(0.1, 0.0)]] * 3
        spec = StackingSpec(stacking_type=StackingType.RADIAL)
        stacking = compute_stacking(spec, n_spans=5)

        with pytest.raises(ValueError):
            apply_stacking_to_sections(sections, stacking)
