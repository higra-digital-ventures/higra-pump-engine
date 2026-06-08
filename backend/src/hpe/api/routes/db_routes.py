"""Database API endpoints for projects and designs persistence.

    GET    /api/v1/projects              — list projects
    POST   /api/v1/projects              — create project
    GET    /api/v1/projects/{id}         — get project
    PUT    /api/v1/projects/{id}         — update project
    DELETE /api/v1/projects/{id}         — delete project + all designs
    GET    /api/v1/projects/{id}/designs — list designs
    POST   /api/v1/projects/{id}/designs — save design
    DELETE /api/v1/designs/{id}          — delete single design
    GET    /api/v1/designs/{id}          — get design
    GET    /api/v1/designs/{id}/curves   — get performance curve
    GET    /api/v1/db/status             — DB health check
"""
from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["database"])


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    machine_type: str = "centrifugal_pump"
    user_id: Optional[str] = None


class DesignSave(BaseModel):
    sizing_result: dict
    operating_point: dict
    overrides: dict = {}
    notes: str = ""
    curve_points: list[dict] = []


# ── DB health ─────────────────────────────────────────────────────────────────

@router.get("/db/status")
def db_status() -> dict:
    try:
        from hpe.db.connection import test_connection
        ok = test_connection()
        return {"status": "ok" if ok else "error", "database": "db_pump_engine"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Projects ──────────────────────────────────────────────────────────────────

@router.get("/projects")
def list_projects_endpoint(user_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    try:
        from hpe.db.repositories import list_projects
        return list_projects(user_id=user_id, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects", status_code=201)
def create_project_endpoint(req: ProjectCreate) -> dict:
    try:
        from hpe.db.repositories import create_project
        return create_project(req.name, req.machine_type, req.description, req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/projects/{project_id}")
def update_project_endpoint(project_id: str, req: ProjectCreate) -> dict:
    try:
        from hpe.db.repositories import update_project
        return update_project(project_id, req.name, req.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}")
def delete_project_endpoint(project_id: str) -> Response:
    try:
        from hpe.db.repositories import delete_project
        found = delete_project(project_id)
        if not found:
            raise HTTPException(status_code=404, detail="Project not found")
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}")
def get_project_endpoint(project_id: str) -> dict:
    try:
        from hpe.db.repositories import get_project
        p = get_project(project_id)
        if not p:
            raise HTTPException(status_code=404, detail="Project not found")
        return p
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Designs ───────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/designs")
def list_designs_endpoint(project_id: str, limit: int = 100) -> list[dict]:
    try:
        from hpe.db.repositories import list_designs
        return list_designs(project_id, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/designs", status_code=201)
def save_design_endpoint(project_id: str, req: DesignSave) -> dict:
    try:
        from hpe.db.repositories import save_design, save_performance_curve
        design = save_design(
            project_id=project_id,
            sizing_result=req.sizing_result,
            op=req.operating_point,
            overrides=req.overrides,
            notes=req.notes,
        )
        if req.curve_points:
            save_performance_curve(design["id"], req.curve_points)
        return design
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/designs/{design_id}")
def delete_design_endpoint(design_id: str) -> Response:
    try:
        from hpe.db.repositories import delete_design
        found = delete_design(design_id)
        if not found:
            raise HTTPException(status_code=404, detail="Design not found")
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/designs/{design_id}")
def get_design_endpoint(design_id: str) -> dict:
    try:
        from hpe.db.repositories import get_design
        d = get_design(design_id)
        if not d:
            raise HTTPException(status_code=404, detail="Design not found")
        return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/designs/{design_id}/curves")
def get_curves_endpoint(design_id: str) -> list[dict]:
    try:
        from hpe.db.repositories import get_performance_curve
        return get_performance_curve(design_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
