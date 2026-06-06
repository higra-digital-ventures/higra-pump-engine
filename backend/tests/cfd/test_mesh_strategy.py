"""Tests for the mesh-strategy resolver (M4 seam)."""

from __future__ import annotations

from hpe.core.enums import DesignMode
from hpe.cfd.mesh.strategy import mesh_strategy_for, plan_mesh, MeshPlan
from hpe.geometry.backend import ParametricBackend, ImplicitBackend


def test_strategy_for_design_modes():
    assert mesh_strategy_for(DesignMode.CLASSIC) == "body_fitted"
    assert mesh_strategy_for(DesignMode.FREE) == "cut_cell"


def test_strategy_for_backend_instance_uses_capabilities():
    assert mesh_strategy_for(ParametricBackend()) == "body_fitted"
    assert mesh_strategy_for(ImplicitBackend()) == "cut_cell"


def test_plan_mesh_classic_is_implemented():
    plan = plan_mesh(DesignMode.CLASSIC)
    assert isinstance(plan, MeshPlan)
    assert plan.strategy == "body_fitted"
    assert plan.mesher == "snappyHexMesh"
    assert plan.implemented is True


def test_plan_mesh_free_is_planned_not_ready():
    plan = plan_mesh(DesignMode.FREE)
    assert plan.strategy == "cut_cell"
    assert plan.implemented is False        # honest: not wired to a real mesher yet
    assert "not wired" in plan.notes.lower()


def test_plan_mesh_to_dict():
    d = plan_mesh(DesignMode.CLASSIC).to_dict()
    assert d["strategy"] == "body_fitted" and d["implemented"] is True
