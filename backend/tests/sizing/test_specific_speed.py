"""Tests for specific speed calculations."""


import pytest

from hpe.sizing.specific_speed import (
    calc_specific_speed,
    calc_type_number,
    classify_impeller_type,
)


class TestCalcSpecificSpeed:
    def test_centrifugal_pump_range(self) -> None:
        """Q=0.05 m3/s, H=30m, 1750 rpm => Nq ~ 25-40 (radial)."""
        ns, nq = calc_specific_speed(0.05, 30.0, 1750)
        assert 20 < nq < 50

    def test_high_ns_pump(self) -> None:
        """High flow, low head => high Nq (mixed or axial)."""
        ns, nq = calc_specific_speed(1.0, 5.0, 1450)
        assert nq > 100

    def test_low_ns_pump(self) -> None:
        """Low flow, high head => low Nq (radial)."""
        ns, nq = calc_specific_speed(0.005, 100.0, 2950)
        assert nq < 25

    def test_ns_equals_nq(self) -> None:
        """Ns and Nq should be equal when using m3/s and m."""
        ns, nq = calc_specific_speed(0.1, 20.0, 1450)
        assert ns == pytest.approx(nq)

    def test_raises_on_zero_flow(self) -> None:
        with pytest.raises(ValueError):
            calc_specific_speed(0.0, 30.0, 1750)

    def test_raises_on_negative_head(self) -> None:
        with pytest.raises(ValueError):
            calc_specific_speed(0.05, -10.0, 1750)

    def test_raises_on_zero_rpm(self) -> None:
        with pytest.raises(ValueError):
            calc_specific_speed(0.05, 30.0, 0)


class TestClassifyImpellerType:
    def test_radial_slow(self) -> None:
        assert classify_impeller_type(15) == "radial_slow"

    def test_radial(self) -> None:
        assert classify_impeller_type(40) == "radial"

    def test_mixed_flow(self) -> None:
        assert classify_impeller_type(100) == "mixed_flow"

    def test_axial(self) -> None:
        assert classify_impeller_type(200) == "axial"


class TestCalcTypeNumber:
    def test_dimensionless(self) -> None:
        """Type number should be positive and order-of-magnitude ~0.1-2."""
        omega_s = calc_type_number(0.05, 30.0, 1750)
        assert 0.05 < omega_s < 3.0
