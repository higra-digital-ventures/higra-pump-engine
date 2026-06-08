"""Mesh-strategy resolution per design mode (M4 seam).

The classic (B-rep) and free (implicit/voxel) geometries need different CFD
meshing approaches:

* **body_fitted** — the mature path: blockMesh / snappyHexMesh / structured
  O-H blade meshing on a clean STEP solid.
* **cut_cell** — for voxel/SDF geometry without a clean B-rep: cut-cell /
  immersed-boundary meshing (e.g. snappyHexMesh from a watertight STL, or an
  immersed-boundary solver).

This module is the integration seam: it maps a design mode / geometry backend
to a :class:`MeshPlan` that ``hpe.cfd.pipeline`` (and the CEM facade) can branch
on. The cut-cell *generation* itself is future work — ``MeshPlan.implemented``
flags whether the chosen mesher is ready, so callers fail loudly instead of
silently producing a bad mesh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class MeshPlan:
    """Which mesher to use for a given geometry, and whether it is ready."""

    strategy: str          # "body_fitted" | "cut_cell"
    mesher: str            # concrete tool, e.g. "snappyHexMesh", "cut_cell_ibm"
    implemented: bool      # is this path wired to a real mesher yet?
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Concrete mesher chosen per strategy, and whether it is implemented today.
_STRATEGY_MESHER = {
    "body_fitted": ("snappyHexMesh", True),
    "cut_cell": ("cut_cell_ibm", False),   # planned (M4+): immersed boundary
}


def mesh_strategy_for(mode_or_backend: Any) -> str:
    """Resolve the mesh strategy string from a DesignMode, name, or backend."""
    # A backend instance/class exposes capabilities.mesh_strategy directly.
    caps = getattr(mode_or_backend, "capabilities", None)
    if caps is not None and getattr(caps, "mesh_strategy", None):
        return caps.mesh_strategy

    from hpe.geometry.backend import get_backend
    return get_backend(mode_or_backend).capabilities.mesh_strategy


def plan_mesh(mode_or_backend: Any, artifact: Optional[dict] = None) -> MeshPlan:
    """Build a :class:`MeshPlan` for the given mode/backend.

    ``artifact`` (a geometry artifact dict) may refine the plan in the future
    (e.g. choosing refinement levels from voxel resolution); it is accepted now
    so callers can pass it without an API change later.
    """
    strategy = mesh_strategy_for(mode_or_backend)
    mesher, implemented = _STRATEGY_MESHER.get(strategy, ("unknown", False))
    notes = (
        "Body-fitted meshing via the existing snappyHexMesh / structured pipeline."
        if strategy == "body_fitted"
        else "Cut-cell / immersed-boundary meshing for voxel geometry — not wired yet."
    )
    return MeshPlan(strategy=strategy, mesher=mesher, implemented=implemented, notes=notes)
