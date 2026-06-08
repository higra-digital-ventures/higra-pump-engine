"""Tests for compressible flow models."""

from __future__ import annotations

import math

import pytest

from hpe.physics.compressible import (
    AIR,
    CO2,
    R134A,
    CompressibleTriangle,
    FluidState,
    choking_mass_flow,
    compressible_triangle,
    compute_fluid_state,
    isentropic_efficiency,
    mach_number,
    pressure_ratio,
    speed_of_sound,
    stagnation_pressure,
    stagnation_temperature,
    static_from_stagnation,
)


class TestSpeedOfSound:
    def test_air_at_288K(self) -> None:
        a = speed_of_sound(288.0)
        assert abs(a - 340.3) < 1.0  # ~340 m/s at 15°C

    def test_increases_with_temperature(self) -> None:
        a1 = speed_of_sound(250.0)
        a2 = speed_of_sound(350.0)
        assert a2 > a1


class TestMachNumber:
    def test_subsonic(self) -> None:
        M = mach_number(170.0, 288.0)
        assert M < 1.0
        assert M == pytest.approx(0.5, abs=0.02)

    def test_supersonic(self) -> None:
        M = mach_number(500.0, 288.0)
        assert M > 1.0


class TestStagnationProperties:
    def test_T0_greater_than_T(self) -> None:
        T0 = stagnation_temperature(288.0, 100.0)
        assert T0 > 288.0

    def test_T0_at_rest(self) -> None:
        T0 = stagnation_temperature(288.0, 0.0)
        assert T0 == pytest.approx(288.0)

    def test_p0_greater_than_p(self) -> None:
        T = 288.0
        V = 100.0
        T0 = stagnation_temperature(T, V)
        p0 = stagnation_pressure(101325.0, T, T0)
        assert p0 > 101325.0

    def test_isentropic_consistency(self) -> None:
        # p0/p = (T0/T)^(gamma/(gamma-1))
        T = 300.0
        V = 200.0
        T0 = stagnation_temperature(T, V)
        p0 = stagnation_pressure(100000.0, T, T0)
        # Reverse: static from stagnation
        T_back, p_back, _ = static_from_stagnation(T0, p0, V)
        assert T_back == pytest.approx(T, rel=1e-4)
        assert p_back == pytest.approx(100000.0, rel=1e-3)


class TestFluidState:
    def test_complete_state(self) -> None:
        state = compute_fluid_state(T0=300.0, p0=200000.0, V=150.0)
        assert isinstance(state, FluidState)
        assert state.T < state.T0
        assert state.p < state.p0
        assert state.M > 0
        assert state.a > 0
        assert state.rho > 0

    def test_zero_velocity(self) -> None:
        state = compute_fluid_state(T0=300.0, p0=101325.0, V=0.0)
        assert state.T == pytest.approx(state.T0)
        assert state.M == pytest.approx(0.0)


class TestCompressibleTriangle:
    def test_basic_triangle(self) -> None:
        tri = compressible_triangle(
            u=300.0, cm=100.0, cu=250.0,
            T0=300.0, p0=200000.0,
        )
        assert isinstance(tri, CompressibleTriangle)
        assert tri.M_abs > 0
        assert tri.M_rel > 0
        assert tri.static.T > 0

    def test_velocity_consistency(self) -> None:
        tri = compressible_triangle(u=200.0, cm=80.0, cu=150.0, T0=300.0, p0=101325.0)
        c = math.sqrt(80**2 + 150**2)
        assert tri.c == pytest.approx(c)
        wu = 200 - 150
        w = math.sqrt(80**2 + wu**2)
        assert tri.w == pytest.approx(w)


class TestIsentropicEfficiency:
    def test_compressor(self) -> None:
        # Compressor: PR=2, T01=300K, T02=370K
        eta = isentropic_efficiency(
            T01=300.0, T02=370.0,
            p01=100000.0, p02=200000.0,
            is_compressor=True,
        )
        assert 0.7 < eta < 1.0

    def test_turbine(self) -> None:
        eta = isentropic_efficiency(
            T01=1200.0, T02=900.0,
            p01=800000.0, p02=100000.0,
            is_compressor=False,
        )
        assert 0.5 < eta < 1.0

    def test_perfect_efficiency(self) -> None:
        # If T02 = T02s (isentropic process)
        T01 = 300.0
        p01 = 100000.0
        p02 = 200000.0
        exp = (AIR.gamma - 1.0) / AIR.gamma
        T02s = T01 * (p02 / p01)**exp
        eta = isentropic_efficiency(T01, T02s, p01, p02, is_compressor=True)
        assert eta == pytest.approx(1.0, abs=0.01)


class TestPressureRatio:
    def test_compressor_pr(self) -> None:
        pr = pressure_ratio(T01=300.0, T02=400.0, eta_s=0.85, is_compressor=True)
        assert pr > 1.0

    def test_turbine_pr(self) -> None:
        pr = pressure_ratio(T01=1200.0, T02=900.0, eta_s=0.90, is_compressor=False)
        assert pr < 1.0


class TestChokingFlow:
    def test_positive(self) -> None:
        m_dot = choking_mass_flow(A=0.01, p0=200000.0, T0=300.0)
        assert m_dot > 0

    def test_scales_with_area(self) -> None:
        m1 = choking_mass_flow(A=0.01, p0=200000.0, T0=300.0)
        m2 = choking_mass_flow(A=0.02, p0=200000.0, T0=300.0)
        assert m2 == pytest.approx(2 * m1, rel=1e-6)

    def test_scales_with_pressure(self) -> None:
        m1 = choking_mass_flow(A=0.01, p0=100000.0, T0=300.0)
        m2 = choking_mass_flow(A=0.01, p0=200000.0, T0=300.0)
        assert m2 == pytest.approx(2 * m1, rel=1e-6)


class TestGasPresets:
    def test_air_properties(self) -> None:
        assert AIR.gamma == pytest.approx(1.4)
        assert AIR.R == pytest.approx(287.05)

    def test_cv_from_cp(self) -> None:
        assert AIR.cv == pytest.approx(AIR.cp - AIR.R)

    def test_multiple_gases(self) -> None:
        from hpe.physics.compressible import NITROGEN
        for gas in [AIR, CO2, NITROGEN, R134A]:
            assert gas.gamma > 1.0
            assert gas.R > 0
            assert gas.cp > 0
