"""Geometry mode routes (M2) — classic vs free, dispatched via GeometryBackend.

Additive endpoints that expose the pluggable geometry backends:

* ``GET  /api/v1/geometry/backends``  — list backends + capabilities (UI uses
  this to enable/disable features per mode).
* ``POST /api/v1/geometry/generate``  — run sizing then generate geometry with
  the backend selected by ``mode`` (``classic`` | ``free``).

The classic path delegates to the existing parametric pipeline; the free path
currently returns HTTP 501 (NotImplementedError) until the implicit kernel
lands. Nothing here changes the legacy ``/api/v1/geometry/*`` routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hpe.core.enums import DesignMode
from hpe.geometry.backend import get_backend, list_backends

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/geometry", tags=["geometry-modes"])


class GenerateRequest(BaseModel):
    Q: float = Field(..., gt=0, description="Flow rate [m³/s]")
    H: float = Field(..., gt=0, description="Head [m]")
    n: float = Field(..., gt=0, description="Rotational speed [rpm]")
    mode: str = Field("classic", description="Design mode: 'classic' | 'free'")
    export: bool = Field(False, description="Attempt STEP/STL export + upload")


@router.get("/backends", summary="List geometry backends and their capabilities")
def backends() -> dict:
    return {"backends": list_backends()}


@router.post("/generate", summary="Generate geometry via the selected design mode")
def generate(req: GenerateRequest) -> dict:
    try:
        mode = DesignMode(req.mode)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown mode {req.mode!r}")

    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing
    from hpe.core.design_state import DesignState

    op = OperatingPoint(flow_rate=req.Q, head=req.H, rpm=req.n)
    sizing = run_sizing(op)

    backend = get_backend(mode)
    try:
        artifact = backend.generate(sizing, export=req.export)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("geometry.generate error")
        raise HTTPException(status_code=500, detail=f"Geometry error: {exc}")

    state = DesignState.from_operating_point(op, mode=mode)
    state.record_stage("sizing", _sizing_dict(sizing))
    state.record_stage("geometry", artifact.to_dict(), note=backend.name)

    return {
        "mode": mode.value,
        "backend": backend.name,
        "mesh_strategy": artifact.mesh_strategy,
        "artifact": artifact.to_dict(),
        "summary": state.summary(),
    }


def _sizing_dict(r: object) -> dict:
    """Extract the fields DesignState constraint checks need from a SizingResult."""
    return {
        "specific_speed_nq": getattr(r, "specific_speed_nq", None),
        "impeller_d2": getattr(r, "impeller_d2", None),
        "estimated_efficiency": getattr(r, "estimated_efficiency", None),
        "estimated_npsh_r": getattr(r, "estimated_npsh_r", None),
        "sigma": getattr(r, "sigma", None),
        "diffusion_ratio": getattr(r, "diffusion_ratio", 0.0),
        "velocity_triangles": getattr(r, "velocity_triangles", {}) or {},
    }
