"""Tests for impeller sizing calculations."""


from hpe.sizing.impeller_sizing import (
    calc_blade_count,
    calc_d1_d2_ratio,
    calc_head_coefficient,
    calc_outlet_blade_angle,
    calc_outlet_width_ratio,
    size_impeller,
)


class TestHeadCoefficient:
    def test_typical_range(self) -> None:
        """psi should be 0.35-1.3 for centrifugal pumps."""
        psi = calc_head_coefficient(35)
        assert 0.35 <= psi <= 1.3

    def test_decreases_with_nq(self) -> None:
        psi_low = calc_head_coefficient(20)
        psi_high = calc_head_coefficient(80)
        assert psi_low > psi_high


class TestD1D2Ratio:
    def test_range(self) -> None:
        ratio = calc_d1_d2_ratio(35)
        assert 0.30 <= ratio <= 0.80

    def test_increases_with_nq(self) -> None:
        ratio_low = calc_d1_d2_ratio(20)
        ratio_high = calc_d1_d2_ratio(100)
        assert ratio_high > ratio_low


class TestBladeCount:
    def test_centrifugal_range(self) -> None:
        """Centrifugal pumps: typically 5-9 blades."""
        z = calc_blade_count(0.25, 0.10, 20, 25)
        assert 5 <= z <= 12

    def test_integer(self) -> None:
        z = calc_blade_count(0.25, 0.10, 20, 25)
        assert isinstance(z, int)


class TestOutletWidthRatio:
    def test_increases_with_nq(self) -> None:
        r_low = calc_outlet_width_ratio(20)
        r_high = calc_outlet_width_ratio(80)
        assert r_high > r_low


class TestOutletBladeAngle:
    def test_range(self) -> None:
        beta2 = calc_outlet_blade_angle(35)
        assert 15 <= beta2 <= 40

    def test_increases_with_nq(self) -> None:
        b_low = calc_outlet_blade_angle(20)
        b_high = calc_outlet_blade_angle(80)
        assert b_high > b_low


class TestSizeImpeller:
    def test_centrifugal_pump(self) -> None:
        """Q=0.05 m3/s, H=30m, 1750 rpm => D2 ~ 200-350mm."""
        imp = size_impeller(0.05, 30.0, 1750, 35.0, 0.82)
        assert 0.15 < imp.d2 < 0.40
        assert imp.d1 < imp.d2
        assert imp.b2 > 0
        assert imp.b1 >= imp.b2
        assert 5 <= imp.blade_count <= 12
        assert 10 < imp.beta1 < 60
        assert 15 < imp.beta2 < 40

    def test_d1_less_than_d2(self) -> None:
        imp = size_impeller(0.05, 30.0, 1750, 35.0, 0.82)
        assert imp.d1 < imp.d2

    def test_hub_smaller_than_d1(self) -> None:
        imp = size_impeller(0.05, 30.0, 1750, 35.0, 0.82)
        assert imp.d1_hub < imp.d1
