"""Tests for LE/TE elliptical modification."""

from __future__ import annotations


import pytest

from hpe.geometry.models import BladeProfile, RunnerGeometryParams
from hpe.geometry.runner.blade import generate_blade_profile
from hpe.geometry.runner.lete_modification import (
    LETEResult,
    LETESpec,
    apply_lete_modification,
    calc_edge_radius,
)


@pytest.fixture
def blade_profile() -> BladeProfile:
    params = RunnerGeometryParams(
        d2=0.32, d1=0.16, d1_hub=0.06,
        b2=0.025, b1=0.035,
        beta1=22.0, beta2=25.0,
        blade_count=7, blade_thickness=0.004,
    )
    return generate_blade_profile(params, n_points=50)


class TestEdgeRadius:
    def test_positive(self) -> None:
        r = calc_edge_radius(0.004, 2.0)
        assert r > 0

    def test_higher_ratio_larger_radius(self) -> None:
        r1 = calc_edge_radius(0.004, 1.5)
        r2 = calc_edge_radius(0.004, 3.0)
        assert r2 > r1

    def test_zero_thickness(self) -> None:
        r = calc_edge_radius(0.0, 2.0)
        assert r == 0.0


class TestLETEModification:
    def test_basic_modification(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec()
        result = apply_lete_modification(blade_profile, spec)
        assert isinstance(result, LETEResult)
        assert len(result.profile.pressure_side) == 50
        assert len(result.profile.suction_side) == 50

    def test_le_thinner_than_midchord(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec(le_enabled=True, te_enabled=False)
        result = apply_lete_modification(blade_profile, spec)

        # LE should be thinner than the midchord region
        mid = len(result.profile.pressure_side) // 2
        le_gap = abs(result.profile.pressure_side[0][1] - result.profile.suction_side[0][1])
        mid_gap = abs(result.profile.pressure_side[mid][1] - result.profile.suction_side[mid][1])
        assert le_gap < mid_gap

    def test_te_thinner_than_midchord(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec(le_enabled=False, te_enabled=True)
        result = apply_lete_modification(blade_profile, spec)

        mid = len(result.profile.pressure_side) // 2
        te_gap = abs(result.profile.pressure_side[-1][1] - result.profile.suction_side[-1][1])
        mid_gap = abs(result.profile.pressure_side[mid][1] - result.profile.suction_side[mid][1])
        assert te_gap < mid_gap

    def test_disabled_returns_original(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec(le_enabled=False, te_enabled=False)
        result = apply_lete_modification(blade_profile, spec)
        # Should be identical to original
        for i in range(len(blade_profile.pressure_side)):
            assert result.profile.pressure_side[i] == blade_profile.pressure_side[i]
            assert result.profile.suction_side[i] == blade_profile.suction_side[i]

    def test_edge_radius_computed(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec(le_elliptic_ratio=2.5, te_elliptic_ratio=1.8)
        result = apply_lete_modification(blade_profile, spec)
        assert result.le_radius >= 0
        assert result.te_radius >= 0

    def test_filing_reduces_thickness(self, blade_profile: BladeProfile) -> None:
        spec_no_file = LETESpec(le_filing_ratio=0.0)
        spec_file = LETESpec(le_filing_ratio=0.5)
        r_no = apply_lete_modification(blade_profile, spec_no_file)
        r_file = apply_lete_modification(blade_profile, spec_file)
        assert r_file.le_thickness <= r_no.le_thickness + 1e-8

    def test_larger_extent_affects_more(self, blade_profile: BladeProfile) -> None:
        spec_small = LETESpec(le_extent=0.05)
        spec_large = LETESpec(le_extent=0.25)
        r_small = apply_lete_modification(blade_profile, spec_small)
        r_large = apply_lete_modification(blade_profile, spec_large)
        # With larger extent, more points should differ from original
        diffs_small = sum(
            1 for i in range(len(blade_profile.pressure_side))
            if abs(r_small.profile.pressure_side[i][1] - blade_profile.pressure_side[i][1]) > 1e-10
        )
        diffs_large = sum(
            1 for i in range(len(blade_profile.pressure_side))
            if abs(r_large.profile.pressure_side[i][1] - blade_profile.pressure_side[i][1]) > 1e-10
        )
        assert diffs_large >= diffs_small

    def test_min_thickness_enforced(self, blade_profile: BladeProfile) -> None:
        spec = LETESpec(min_edge_thickness=0.001)
        result = apply_lete_modification(blade_profile, spec)
        # LE and TE thickness should be at least min
        assert result.le_thickness >= 0.001 - 1e-8
        assert result.te_thickness >= 0.001 - 1e-8
