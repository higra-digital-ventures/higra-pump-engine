"""Tests for advanced loss models."""

from __future__ import annotations

import math

import pytest

from hpe.core.models import OperatingPoint
from hpe.physics.advanced_losses import (
    AdvancedLossBreakdown,
    calc_advanced_losses,
    calc_endwall_loss,
    calc_mixing_loss,
    calc_profile_loss,
    calc_tip_leakage_loss,
)
from hpe.sizing.meanline import run_sizing
from hpe.sizing.velocity_triangles import calc_inlet_triangle, calc_outlet_triangle


@pytest.fixture
def sizing_result():
    op = OperatingPoint(flow_rate=0.05, head=30.0, rpm=1750)
    return run_sizing(op)


@pytest.fixture
def triangles(sizing_result):
    sr = sizing_result
    mp = sr.meridional_profile
    u2 = sr.velocity_triangles["outlet"]["u"]
    rpm = 60.0 * u2 / (math.pi * sr.impeller_d2)

    tri_in = calc_inlet_triangle(
        d1=sr.impeller_d1, b1=mp.get("b1", sr.impeller_b2 * 1.2),
        flow_rate=0.05, rpm=rpm,
    )
    tri_out = calc_outlet_triangle(
        d2=sr.impeller_d2, b2=sr.impeller_b2,
        flow_rate=0.05, rpm=rpm,
        beta2=sr.beta2, blade_count=sr.blade_count,
    )
    return tri_in, tri_out


class TestProfileLoss:
    def test_positive_values(self, triangles) -> None:
        tri_in, tri_out = triangles
        ps, ss, total = calc_profile_loss(
            tri_in, tri_out, 0.16, 0.32, 0.025, 7,
        )
        assert ps > 0
        assert ss > 0
        assert total == pytest.approx(ps + ss)

    def test_ss_greater_than_ps(self, triangles) -> None:
        tri_in, tri_out = triangles
        ps, ss, _ = calc_profile_loss(
            tri_in, tri_out, 0.16, 0.32, 0.025, 7,
        )
        assert ss > ps  # Suction side always has more loss

    def test_roughness_increases_loss(self, triangles) -> None:
        tri_in, tri_out = triangles
        _, _, loss_smooth = calc_profile_loss(
            tri_in, tri_out, 0.16, 0.32, 0.025, 7,
            surface_roughness=1e-6,
        )
        _, _, loss_rough = calc_profile_loss(
            tri_in, tri_out, 0.16, 0.32, 0.025, 7,
            surface_roughness=50e-6,
        )
        assert loss_rough >= loss_smooth


class TestTipLeakage:
    def test_positive(self, triangles) -> None:
        _, tri_out = triangles
        loss = calc_tip_leakage_loss(tri_out, 0.32, 0.025, 7)
        assert loss > 0

    def test_larger_clearance_more_loss(self, triangles) -> None:
        _, tri_out = triangles
        loss_small = calc_tip_leakage_loss(tri_out, 0.32, 0.025, 7, tip_clearance=0.3e-3)
        loss_large = calc_tip_leakage_loss(tri_out, 0.32, 0.025, 7, tip_clearance=1.0e-3)
        assert loss_large > loss_small

    def test_closed_impeller_less(self, triangles) -> None:
        _, tri_out = triangles
        loss_open = calc_tip_leakage_loss(tri_out, 0.32, 0.025, 7, is_open_impeller=True)
        loss_closed = calc_tip_leakage_loss(tri_out, 0.32, 0.025, 7, is_open_impeller=False)
        assert loss_closed < loss_open


class TestEndwallLoss:
    def test_positive(self, triangles) -> None:
        tri_in, tri_out = triangles
        hub, shroud, total = calc_endwall_loss(
            tri_in, tri_out, 0.16, 0.32, 0.035, 0.025, 7,
        )
        assert hub > 0
        assert shroud > 0
        assert total == pytest.approx(hub + shroud)

    def test_shroud_greater_than_hub(self, triangles) -> None:
        tri_in, tri_out = triangles
        hub, shroud, _ = calc_endwall_loss(
            tri_in, tri_out, 0.16, 0.32, 0.035, 0.025, 7,
        )
        assert shroud > hub  # Stationary shroud sees higher relative velocity


class TestMixingLoss:
    def test_positive(self, triangles) -> None:
        _, tri_out = triangles
        loss = calc_mixing_loss(tri_out, 0.025, 7)
        assert loss > 0

    def test_thicker_blade_more_loss(self, triangles) -> None:
        _, tri_out = triangles
        loss_thin = calc_mixing_loss(tri_out, 0.025, 7, blade_thickness=0.002)
        loss_thick = calc_mixing_loss(tri_out, 0.025, 7, blade_thickness=0.008)
        assert loss_thick > loss_thin


class TestAdvancedLossTotal:
    def test_full_breakdown(self, sizing_result, triangles) -> None:
        tri_in, tri_out = triangles
        result = calc_advanced_losses(
            sizing_result, q_actual=0.05, q_design=0.05,
            tri_in=tri_in, tri_out=tri_out,
        )
        assert isinstance(result, AdvancedLossBreakdown)
        assert result.total_head_loss > 0
        assert result.loss_coefficient > 0
        assert result.loss_coefficient < 1.0  # Should be fraction of Euler head

    def test_loss_breakdown_sums(self, sizing_result, triangles) -> None:
        tri_in, tri_out = triangles
        r = calc_advanced_losses(
            sizing_result, q_actual=0.05, q_design=0.05,
            tri_in=tri_in, tri_out=tri_out,
        )
        expected_total = (
            r.profile_loss_total + r.tip_leakage + r.endwall_total
            + r.mixing + r.incidence + r.recirculation
        )
        assert r.total_head_loss == pytest.approx(expected_total, rel=1e-6)

    def test_off_design_higher_loss(self, sizing_result, triangles) -> None:
        tri_in, tri_out = triangles
        loss_design = calc_advanced_losses(
            sizing_result, q_actual=0.05, q_design=0.05,
            tri_in=tri_in, tri_out=tri_out,
        )
        # At part-load, recirculation kicks in
        from hpe.sizing.velocity_triangles import calc_inlet_triangle, calc_outlet_triangle

        sr = sizing_result
        mp = sr.meridional_profile
        u2 = sr.velocity_triangles["outlet"]["u"]
        rpm = 60.0 * u2 / (math.pi * sr.impeller_d2)

        tri_in_off = calc_inlet_triangle(
            d1=sr.impeller_d1, b1=mp.get("b1", sr.impeller_b2 * 1.2),
            flow_rate=0.025, rpm=rpm,
        )
        tri_out_off = calc_outlet_triangle(
            d2=sr.impeller_d2, b2=sr.impeller_b2,
            flow_rate=0.025, rpm=rpm,
            beta2=sr.beta2, blade_count=sr.blade_count,
        )
        loss_off = calc_advanced_losses(
            sizing_result, q_actual=0.025, q_design=0.05,
            tri_in=tri_in_off, tri_out=tri_out_off,
        )
        assert loss_off.total_head_loss > loss_design.total_head_loss
