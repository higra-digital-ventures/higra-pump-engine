"""Tests for the specific-speed-aware blade count (calc_blade_count)."""

from __future__ import annotations

from hpe.sizing.impeller_sizing import calc_blade_count


# Representative radial geometry (D2=300mm, eye 150mm, typical angles).
_D2, _D1, _B1, _B2 = 0.30, 0.15, 18.0, 25.0


def _z(nq):
    return calc_blade_count(_D2, _D1, _B1, _B2, nq)


def test_radial_regime_uses_pfleiderer_range():
    # Radial pumps: 6-7 blades typically.
    assert 5 <= _z(30) <= 9
    assert 5 <= _z(60) <= 9


def test_blade_count_decreases_into_mixed_flow():
    """Physical expectation: fewer blades as specific speed rises."""
    assert _z(97) <= 6           # the previously-xfailed case
    assert _z(120) <= _z(60)     # monotonic non-increasing into mixed regime


def test_axial_regime_few_blades():
    assert _z(180) <= 4
    assert _z(220) <= 4
    assert _z(220) >= 3


def test_monotonic_non_increasing_with_nq():
    counts = [_z(nq) for nq in (30, 80, 100, 140, 180, 220)]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_nq_none_keeps_legacy_radial_behaviour():
    # Without nq, behaves as the legacy Pfleiderer clamp (>= BLADE_COUNT_MIN).
    assert _z(None) >= 5
