"""Tests for the implicit SDF geometry prototype (M3)."""

from __future__ import annotations

import numpy as np
import pytest

from hpe.geometry.implicit import sdf as S
from hpe.geometry.implicit.field import sample, ScalarField
from hpe.geometry.implicit.builder import build_impeller_sdf


def _grid(n=21, span=2.0):
    xs = np.linspace(-span, span, n)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    return X, Y, Z


# ---------------------------------------------------------------------------
# SDF primitives: sign convention (negative inside, positive outside)
# ---------------------------------------------------------------------------

def test_sphere_sign_convention():
    f = S.sphere(1.0)
    assert f(np.array(0.0), np.array(0.0), np.array(0.0)) < 0      # centre inside
    assert f(np.array(2.0), np.array(0.0), np.array(0.0)) > 0      # far outside
    assert f(np.array(1.0), np.array(0.0), np.array(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_box_inside_outside():
    f = S.box((1.0, 1.0, 1.0))
    assert f(np.array(0.0), np.array(0.0), np.array(0.0)) < 0
    assert f(np.array(2.0), np.array(0.0), np.array(0.0)) > 0


def test_capped_cylinder_axial_caps():
    f = S.capped_cylinder_z(1.0, 0.0, 2.0)
    assert f(np.array(0.0), np.array(0.0), np.array(1.0)) < 0      # inside
    assert f(np.array(0.0), np.array(0.0), np.array(3.0)) > 0      # above top cap
    assert f(np.array(0.0), np.array(0.0), np.array(-1.0)) > 0     # below bottom cap


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------

def test_union_and_subtract():
    a = S.sphere(1.0, center=(-0.5, 0, 0))
    b = S.sphere(1.0, center=(0.5, 0, 0))
    u = S.union(a, b)
    # A point inside either sphere is inside the union.
    assert u(np.array(-1.3), np.array(0.0), np.array(0.0)) < 0

    diff = S.subtract(a, b)
    # A point inside b (the cutter) must be removed from a.
    assert diff(np.array(0.5), np.array(0.0), np.array(0.0)) > 0


def test_rotate_z_preserves_distance_for_symmetric():
    f = S.capped_cylinder_z(1.0, 0.0, 2.0)
    g = S.rotate_z(f, np.pi / 3)
    p = (np.array(0.4), np.array(0.0), np.array(1.0))
    assert f(*p) < 0 and g(*p) < 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_produces_field_with_inside_voxels():
    f = S.sphere(1.0)
    field = sample(f, ((-2, 2), (-2, 2), (-2, 2)), resolution=21)
    assert isinstance(field, ScalarField)
    assert field.values.shape == (21, 21, 21)
    assert field.inside_count > 0
    assert field.solid_volume > 0
    # Sampled sphere volume should be in the right ballpark (4/3 π ≈ 4.19).
    assert 3.0 < field.solid_volume < 5.5


# ---------------------------------------------------------------------------
# Impeller builder
# ---------------------------------------------------------------------------

class _FakeSizing:
    impeller_d2 = 0.30
    impeller_d1 = 0.13
    impeller_b2 = 0.02
    blade_count = 6
    blade_thickness = 0.006


def test_build_impeller_sdf_meta_and_solidity():
    solid, meta, bounds = build_impeller_sdf(_FakeSizing())
    assert meta["blade_count"] == 6
    assert meta["R2_mm"] == pytest.approx(150.0, abs=0.1)
    field = sample(solid, bounds, resolution=32)
    assert field.inside_count > 0       # the disk+blades enclose volume
