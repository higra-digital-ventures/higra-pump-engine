"""Tests for hpe.geometry.backend — pluggable geometry backends (M1)."""

from __future__ import annotations

import pytest

from hpe.core.enums import DesignMode
from hpe.geometry.backend import (
    GeometryBackend, GeometryArtifact, ParametricBackend, ImplicitBackend,
    get_backend, list_backends, register_backend,
)


# ---------------------------------------------------------------------------
# registry resolution
# ---------------------------------------------------------------------------

def test_get_backend_by_design_mode():
    assert get_backend(DesignMode.CLASSIC).name == "parametric"
    assert get_backend(DesignMode.FREE).name == "implicit"


def test_get_backend_by_string_value_and_name():
    assert get_backend("classic").name == "parametric"   # DesignMode value
    assert get_backend("parametric").name == "parametric"  # backend name
    assert get_backend("free").name == "implicit"


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend("does-not-exist")


def test_list_backends_shape():
    backends = {b["name"]: b for b in list_backends()}
    assert backends["parametric"]["mode"] == "classic"
    assert backends["implicit"]["mode"] == "free"
    assert backends["parametric"]["capabilities"]["mesh_strategy"] == "body_fitted"
    assert backends["implicit"]["capabilities"]["supports_voxel"] is True


# ---------------------------------------------------------------------------
# capabilities + protocol conformance
# ---------------------------------------------------------------------------

def test_capabilities_differ_between_backends():
    classic = ParametricBackend.capabilities
    free = ImplicitBackend.capabilities
    assert classic.supports_step and not classic.supports_voxel
    assert free.supports_voxel and free.supports_internal_channels
    assert classic.mesh_strategy == "body_fitted"
    assert free.mesh_strategy == "cut_cell"


def test_backends_conform_to_protocol():
    assert isinstance(ParametricBackend(), GeometryBackend)
    assert isinstance(ImplicitBackend(), GeometryBackend)


# ---------------------------------------------------------------------------
# implicit backend (M3) — produces a voxel field artifact
# ---------------------------------------------------------------------------

def test_implicit_backend_generates_voxel_artifact(sizing_result):
    art = ImplicitBackend().generate(sizing_result, export=False)
    assert art.backend == "implicit"
    assert art.mesh_strategy == "cut_cell"
    # Voxel statistics are always produced (numpy only).
    vox = art.extra["voxel"]
    assert vox["inside_voxels"] > 0
    assert len(vox["grid_shape"]) == 3
    assert art.params["blade_count"] >= 1


# ---------------------------------------------------------------------------
# parametric backend wraps the existing pipeline (integration, no CadQuery)
# ---------------------------------------------------------------------------

def test_parametric_backend_generates_artifact(sizing_result):
    art = ParametricBackend().generate(sizing_result, export=False)
    assert isinstance(art, GeometryArtifact)
    assert art.backend == "parametric"
    assert art.mesh_strategy == "body_fitted"
    # The parametric pipeline always returns 2D profiles even without CadQuery.
    assert art.params and "D2_mm" in art.params
    assert "hub_r_mm" in art.meridional or art.meridional  # non-empty channel
    # No URLs without export.
    assert art.step_url is None and art.stl_url is None


def test_parametric_backend_to_dict_roundtrip(sizing_result):
    art = ParametricBackend().generate(sizing_result, export=False)
    d = art.to_dict()
    assert d["backend"] == "parametric"
    assert "params" in d and "warnings" in d


def test_register_backend_override():
    class Dummy:
        name = "parametric"
        capabilities = ParametricBackend.capabilities
        def generate(self, sizing, **kw):  # noqa: D401
            return GeometryArtifact(backend="parametric")
    original = get_backend("parametric")
    try:
        register_backend(Dummy())
        assert isinstance(get_backend("parametric"), Dummy)
    finally:
        register_backend(original)  # restore for other tests
