"""Design version history routes — auto-save, restore, compare."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["versions"])

# ---------------------------------------------------------------------------
# In-memory store (no DB required)
# ---------------------------------------------------------------------------
_versions: dict[str, dict[str, Any]] = {}
_counter: int = 0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class OperatingPointIn(BaseModel):
    flow_rate: float
    head: float
    rpm: float


class VersionCreateRequest(BaseModel):
    project_id: str | None = None
    operating_point: OperatingPointIn
    sizing_result: dict[str, Any]
    label: str | None = None


class VersionSummary(BaseModel):
    id: str
    version_number: int
    label: str
    nq: float
    eta: float
    d2_mm: float
    npsh: float
    power_kw: float
    flow_rate: float
    head: float
    rpm: float
    created_at: str


class VersionDetail(BaseModel):
    id: str
    version_number: int
    label: str
    created_at: str
    operating_point: OperatingPointIn
    sizing_result: dict[str, Any]


class CompareRequest(BaseModel):
    version_a: str
    version_b: str


class CompareResponse(BaseModel):
    a: dict[str, Any]
    b: dict[str, Any]
    deltas: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_summary(rec: dict[str, Any]) -> VersionSummary:
    """Build a summary from a stored version record."""
    sr = rec["sizing_result"]
    op = rec["operating_point"]
    return VersionSummary(
        id=rec["id"],
        version_number=rec["version_number"],
        label=rec["label"],
        nq=sr.get("specific_speed_nq", 0),
        eta=sr.get("estimated_efficiency", 0),
        d2_mm=sr.get("impeller_d2", 0) * 1000,
        npsh=sr.get("estimated_npsh_r", 0),
        power_kw=sr.get("estimated_power", 0) / 1000,
        flow_rate=op["flow_rate"],
        head=op["head"],
        rpm=op["rpm"],
        created_at=rec["created_at"],
    )


def _compute_deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compute metric deltas between two sizing results."""
    metrics = [
        ("eta", "estimated_efficiency", 100),  # display as %
        ("d2_mm", "impeller_d2", 1000),
        ("npsh", "estimated_npsh_r", 1),
        ("power_kw", "estimated_power", 0.001),
        ("nq", "specific_speed_nq", 1),
        ("blade_count", "blade_count", 1),
        ("beta1", "beta1", 1),
        ("beta2", "beta2", 1),
    ]
    deltas: dict[str, Any] = {}
    for name, key, scale in metrics:
        va = a.get(key, 0) * scale
        vb = b.get(key, 0) * scale
        diff = vb - va
        pct = (diff / va * 100) if va else 0
        deltas[name] = {"a": round(va, 4), "b": round(vb, 4), "delta": round(diff, 4), "pct": round(pct, 2)}
    return deltas


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/versions", response_model=VersionSummary)
def create_version(req: VersionCreateRequest) -> VersionSummary:
    """Save a new design version."""
    global _counter
    _counter += 1
    vid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    label = req.label or f"V{_counter}"

    rec: dict[str, Any] = {
        "id": vid,
        "version_number": _counter,
        "label": label,
        "created_at": now,
        "project_id": req.project_id,
        "operating_point": req.operating_point.model_dump(),
        "sizing_result": req.sizing_result,
    }
    _versions[vid] = rec
    return _extract_summary(rec)


@router.get("/versions", response_model=list[VersionSummary])
def list_versions(project_id: str | None = None, limit: int = 50) -> list[VersionSummary]:
    """List all versions, newest first."""
    recs = sorted(_versions.values(), key=lambda r: r["created_at"], reverse=True)
    if project_id:
        recs = [r for r in recs if r.get("project_id") == project_id]
    return [_extract_summary(r) for r in recs[:limit]]


@router.get("/versions/{version_id}", response_model=VersionDetail)
def get_version(version_id: str) -> VersionDetail:
    """Get full version details."""
    rec = _versions.get(version_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Version not found")
    return VersionDetail(
        id=rec["id"],
        version_number=rec["version_number"],
        label=rec["label"],
        created_at=rec["created_at"],
        operating_point=OperatingPointIn(**rec["operating_point"]),
        sizing_result=rec["sizing_result"],
    )


@router.post("/versions/compare", response_model=CompareResponse)
def compare_versions(req: CompareRequest) -> CompareResponse:
    """Compare two versions side by side."""
    a = _versions.get(req.version_a)
    b = _versions.get(req.version_b)
    if not a:
        raise HTTPException(status_code=404, detail=f"Version {req.version_a} not found")
    if not b:
        raise HTTPException(status_code=404, detail=f"Version {req.version_b} not found")

    deltas = _compute_deltas(a["sizing_result"], b["sizing_result"])
    return CompareResponse(
        a={"version": _extract_summary(a).model_dump(), "sizing_result": a["sizing_result"]},
        b={"version": _extract_summary(b).model_dump(), "sizing_result": b["sizing_result"]},
        deltas=deltas,
    )


@router.delete("/versions/{version_id}")
def delete_version(version_id: str) -> dict[str, str]:
    """Delete a version."""
    if version_id not in _versions:
        raise HTTPException(status_code=404, detail="Version not found")
    del _versions[version_id]
    return {"status": "deleted", "id": version_id}
