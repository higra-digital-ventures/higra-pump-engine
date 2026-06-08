"""Tests for inverse blade design module."""

from __future__ import annotations


import pytest

from hpe.geometry.inverse.loading import (
    compute_loading_derivative,
    compute_rvt_distribution,
    compute_spanwise_rvt,
)
from hpe.geometry.inverse.models import (
    BladeLoadingSpec,
    InverseDesignSpec,
    LoadingType,
    StackingCondition,
)
from hpe.geometry.inverse.solver import (
    inverse_design,
    inverse_design_to_blade_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_spec() -> InverseDesignSpec:
    """Typical centrifugal pump inverse design spec."""
    loading = BladeLoadingSpec(
        rvt_inlet=0.0,  # No pre-swirl
        rvt_outlet=3.5,  # Typical rVθ at outlet [m²/s]
        loading_type=LoadingType.MID_LOADED,
    )
    return InverseDesignSpec(
        d2=0.32,
        d1=0.16,
        d1_hub=0.06,
        b2=0.025,
        b1=0.035,
        blade_count=7,
        rpm=1750,
        loading=loading,
        n_streamwise=30,
        n_spanwise=3,
    )


# ---------------------------------------------------------------------------
# Loading distribution tests
# ---------------------------------------------------------------------------

class TestLoadingDistribution:
    def test_mid_loaded_endpoints(self) -> None:
        spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        m = [0.0, 0.5, 1.0]
        rvt = compute_rvt_distribution(spec, m)
        assert abs(rvt[0] - 0.0) < 1e-10
        assert abs(rvt[-1] - 4.0) < 1e-10

    def test_mid_loaded_symmetry(self) -> None:
        spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        m = [0.0, 0.25, 0.5, 0.75, 1.0]
        rvt = compute_rvt_distribution(spec, m)
        # Mid-loaded should have ~half at midpoint
        assert abs(rvt[2] - 2.0) < 0.1

    def test_fore_loaded_front_heavy(self) -> None:
        spec = BladeLoadingSpec(
            rvt_inlet=0.0, rvt_outlet=4.0,
            loading_type=LoadingType.FORE_LOADED,
        )
        m = [0.0, 0.3, 0.7, 1.0]
        rvt = compute_rvt_distribution(spec, m)
        # At m=0.3, fore-loaded should have more than mid-loaded
        mid_spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        rvt_mid = compute_rvt_distribution(mid_spec, m)
        assert rvt[1] > rvt_mid[1]

    def test_aft_loaded_rear_heavy(self) -> None:
        spec = BladeLoadingSpec(
            rvt_inlet=0.0, rvt_outlet=4.0,
            loading_type=LoadingType.AFT_LOADED,
        )
        m = [0.0, 0.3, 0.7, 1.0]
        rvt = compute_rvt_distribution(spec, m)
        mid_spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        rvt_mid = compute_rvt_distribution(mid_spec, m)
        assert rvt[1] < rvt_mid[1]

    def test_custom_loading(self) -> None:
        spec = BladeLoadingSpec(
            rvt_inlet=0.0, rvt_outlet=4.0,
            loading_type=LoadingType.CUSTOM,
            loading_control_points=[(0.5, 0.8)],  # 80% loaded at midchord
        )
        m = [0.0, 0.5, 1.0]
        rvt = compute_rvt_distribution(spec, m)
        assert abs(rvt[1] - 3.2) < 0.1  # 0.8 * 4.0

    def test_derivative_nonzero(self) -> None:
        spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        m = [i / 20 for i in range(21)]
        rvt = compute_rvt_distribution(spec, m)
        drvt = compute_loading_derivative(rvt, m)
        assert all(d >= 0 for d in drvt)  # Monotonically increasing
        assert max(drvt) > 0


class TestSpanwiseDistribution:
    def test_free_vortex_constant(self) -> None:
        spec = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=4.0)
        spans = [0.0, 0.5, 1.0]
        rvt_out = compute_spanwise_rvt(spec, spans)
        assert all(abs(v - 4.0) < 1e-10 for v in rvt_out)

    def test_custom_spanwise(self) -> None:
        spec = BladeLoadingSpec(
            rvt_inlet=0.0, rvt_outlet=4.0,
            stacking=StackingCondition.CUSTOM,
            spanwise_rvt=[(0.0, 3.5), (1.0, 4.5)],
        )
        spans = [0.0, 0.5, 1.0]
        rvt_out = compute_spanwise_rvt(spec, spans)
        assert abs(rvt_out[0] - 3.5) < 1e-10
        assert abs(rvt_out[1] - 4.0) < 0.1
        assert abs(rvt_out[2] - 4.5) < 1e-10


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestInverseSolver:
    def test_basic_solve(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)

        assert len(result.blade_sections) == 3  # 3 spans
        assert len(result.span_fractions) == 3
        assert len(result.beta_inlet) == 3
        assert len(result.beta_outlet) == 3

    def test_blade_section_points(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        for section in result.blade_sections:
            assert len(section) == 30  # n_streamwise
            # Radius should increase from inlet to outlet
            r_first = section[0][0]
            r_last = section[-1][0]
            assert r_last > r_first

    def test_wrap_angle_positive(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        for wa in result.wrap_angles:
            assert wa > 0, "Wrap angle should be positive"

    def test_blade_angles_reasonable(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        for beta in result.beta_inlet:
            assert 5 < abs(beta) < 175, f"Inlet angle {beta} out of range"
        for beta in result.beta_outlet:
            assert 5 < abs(beta) < 175, f"Outlet angle {beta} out of range"

    def test_loading_types_differ(self) -> None:
        base = dict(
            d2=0.32, d1=0.16, d1_hub=0.06, b2=0.025, b1=0.035,
            blade_count=7, rpm=1750, n_streamwise=30, n_spanwise=3,
        )
        results = {}
        for lt in [LoadingType.FORE_LOADED, LoadingType.MID_LOADED, LoadingType.AFT_LOADED]:
            loading = BladeLoadingSpec(rvt_inlet=0.0, rvt_outlet=3.5, loading_type=lt)
            spec = InverseDesignSpec(loading=loading, **base)
            results[lt] = inverse_design(spec)

        # Different loading types should produce different wrap angles
        wa_fore = results[LoadingType.FORE_LOADED].wrap_angles[1]
        wa_mid = results[LoadingType.MID_LOADED].wrap_angles[1]
        wa_aft = results[LoadingType.AFT_LOADED].wrap_angles[1]
        assert wa_fore != wa_mid or wa_mid != wa_aft

    def test_diffusion_ratio_computed(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        assert result.diffusion_ratio > 0

    def test_max_loading_computed(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        assert result.max_blade_loading > 0


class TestInverseToBladeProfile:
    def test_conversion(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        profile = inverse_design_to_blade_profile(result, thickness=0.004)

        assert len(profile.camber_points) == 30
        assert len(profile.pressure_side) == 30
        assert len(profile.suction_side) == 30
        assert profile.thickness == 0.004

    def test_thickness_at_endpoints(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        profile = inverse_design_to_blade_profile(result)

        # At LE and TE, thickness should be ~0
        r_ps, theta_ps = profile.pressure_side[0]
        r_ss, theta_ss = profile.suction_side[0]
        assert abs(theta_ps - theta_ss) < 1e-6  # Zero at LE

    def test_midspan_default(self, default_spec: InverseDesignSpec) -> None:
        result = inverse_design(default_spec)
        profile = inverse_design_to_blade_profile(result, span_index=-1)
        # Should use middle span
        assert len(profile.camber_points) > 0
