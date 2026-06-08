"""Project CRUD API routes — create, list, get, save sizing results."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from hpe.core.database import get_session
from hpe.core.db_models import (
    Project,
    SizingResultRecord,
)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


# --- Schemas ---

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    machine_type: str = Field("centrifugal_pump")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    machine_type: str
    created_at: str
    n_sizing_results: int = 0


class SaveSizingRequest(BaseModel):
    flow_rate: float = Field(..., gt=0)
    head: float = Field(..., gt=0)
    rpm: float = Field(..., gt=0)
    label: str = Field("design")


class SizingRecordResponse(BaseModel):
    id: str
    specific_speed_nq: float
    impeller_d2: float
    impeller_d1: float
    impeller_b2: float
    blade_count: int
    beta1: float
    beta2: float
    estimated_efficiency: float
    estimated_power: float
    created_at: str


# --- Dependency ---

def get_db() -> Session:
    session = get_session()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()


# --- Endpoints ---

@router.post("", response_model=ProjectResponse)
def create_project(req: ProjectCreate, db: Session = Depends(get_db)) -> ProjectResponse:
    """Create a new project."""
    project = Project(
        name=req.name,
        description=req.description,
        machine_type=req.machine_type,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        machine_type=project.machine_type,
        created_at=str(project.created_at),
    )


@router.get("", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    """List all projects."""
    projects = db.query(Project).order_by(Project.updated_at.desc()).all()
    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            machine_type=p.machine_type,
            created_at=str(p.created_at),
            n_sizing_results=len(p.sizing_results),
        )
        for p in projects
    ]


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    """Get a single project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        machine_type=project.machine_type,
        created_at=str(project.created_at),
        n_sizing_results=len(project.sizing_results),
    )


@router.post("/{project_id}/sizing", response_model=SizingRecordResponse)
def save_sizing(
    project_id: str, req: SaveSizingRequest, db: Session = Depends(get_db),
) -> SizingRecordResponse:
    """Run sizing and save result to the project."""
    from hpe.core.models import OperatingPoint
    from hpe.core.persistence import operating_point_to_record, sizing_result_to_record
    from hpe.sizing.meanline import run_sizing

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    result = run_sizing(op)

    # Save operating point
    op_record = operating_point_to_record(op, project_id, label=req.label)
    db.add(op_record)
    db.flush()

    # Save sizing result
    sr_record = sizing_result_to_record(result, project_id, op_record.id)
    db.add(sr_record)
    db.commit()
    db.refresh(sr_record)

    return SizingRecordResponse(
        id=sr_record.id,
        specific_speed_nq=sr_record.specific_speed_nq,
        impeller_d2=sr_record.impeller_d2,
        impeller_d1=sr_record.impeller_d1,
        impeller_b2=sr_record.impeller_b2,
        blade_count=sr_record.blade_count,
        beta1=sr_record.beta1,
        beta2=sr_record.beta2,
        estimated_efficiency=sr_record.estimated_efficiency,
        estimated_power=sr_record.estimated_power,
        created_at=str(sr_record.created_at),
    )


@router.get("/{project_id}/sizing", response_model=List[SizingRecordResponse])
def list_sizing_results(
    project_id: str, db: Session = Depends(get_db),
) -> list[SizingRecordResponse]:
    """List all sizing results for a project."""
    records = (
        db.query(SizingResultRecord)
        .filter(SizingResultRecord.project_id == project_id)
        .order_by(SizingResultRecord.created_at.desc())
        .all()
    )
    return [
        SizingRecordResponse(
            id=r.id,
            specific_speed_nq=r.specific_speed_nq,
            impeller_d2=r.impeller_d2,
            impeller_d1=r.impeller_d1,
            impeller_b2=r.impeller_b2,
            blade_count=r.blade_count,
            beta1=r.beta1,
            beta2=r.beta2,
            estimated_efficiency=r.estimated_efficiency,
            estimated_power=r.estimated_power,
            created_at=str(r.created_at),
        )
        for r in records
    ]
