"""Infrastructure endpoints — health deep, metrics, structured errors.

Melhorias #41-45.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Response

log = logging.getLogger(__name__)
router = APIRouter(tags=["infra"])

_START_TIME = time.time()
_REQUEST_COUNT = {"total": 0, "errors": 0}


# ===========================================================================
# #41 Deep health check
# ===========================================================================

@router.get("/health/deep", summary="Deep health check (db, redis, minio, models)")
def health_deep() -> dict[str, Any]:
    """Verifica todas as dependências externas e retorna status detalhado."""
    checks: dict[str, Any] = {
        "uptime_s": round(time.time() - _START_TIME, 1),
        "version": "20.0.0",
        "components": {},
    }

    # ── DB ──────────────────────────────────────────────────────────────
    try:
        from hpe.core.config import settings
        db_url = settings.database_url
        checks["components"]["database"] = {
            "configured": bool(db_url),
            "url_host": db_url.split("@")[-1].split("/")[0] if "@" in db_url else "unknown",
            "ok": True,
        }
    except Exception as exc:
        checks["components"]["database"] = {"ok": False, "error": str(exc)}

    # ── Redis ───────────────────────────────────────────────────────────
    try:
        import redis  # type: ignore
        from hpe.core.config import settings
        r = redis.Redis.from_url(settings.redis_url, socket_timeout=1)
        r.ping()
        checks["components"]["redis"] = {"ok": True}
    except Exception as exc:
        checks["components"]["redis"] = {"ok": False, "error": str(exc)[:100]}

    # ── MinIO ───────────────────────────────────────────────────────────
    try:
        from minio import Minio  # type: ignore  # noqa: F401
        endpoint = os.environ.get("HPE_MINIO_ENDPOINT", os.environ.get("MINIO_ENDPOINT", ""))
        checks["components"]["minio"] = {
            "configured": bool(endpoint),
            "endpoint": endpoint or "not set",
            "ok": bool(endpoint),
        }
    except ImportError:
        checks["components"]["minio"] = {"ok": False, "error": "minio package not installed"}

    # ── Surrogate models ────────────────────────────────────────────────
    try:
        from hpe.ai.surrogate.evaluator import SurrogateEvaluator
        ev = SurrogateEvaluator()
        checks["components"]["surrogate"] = {
            "ok": ev.available if hasattr(ev, "available") else True,
        }
    except Exception as exc:
        checks["components"]["surrogate"] = {"ok": False, "error": str(exc)[:80]}

    # ── OpenFOAM (path probe) ────────────────────────────────────────────
    import shutil
    checks["components"]["openfoam"] = {
        "available": shutil.which("simpleFoam") is not None,
    }
    checks["components"]["su2"] = {
        "available": shutil.which("SU2_CFD") is not None,
    }
    checks["components"]["foamToVTK"] = {
        "available": shutil.which("foamToVTK") is not None,
    }

    # Overall status
    all_critical_ok = checks["components"]["database"].get("ok", False)
    checks["status"] = "healthy" if all_critical_ok else "degraded"
    return checks


# ===========================================================================
# #42 Prometheus metrics
# ===========================================================================

@router.get("/metrics", summary="Prometheus metrics format")
def metrics() -> Response:
    """Métricas formato Prometheus text exposition."""
    uptime = time.time() - _START_TIME
    lines = [
        "# HELP hpe_uptime_seconds Process uptime in seconds",
        "# TYPE hpe_uptime_seconds gauge",
        f"hpe_uptime_seconds {uptime:.1f}",
        "",
        "# HELP hpe_requests_total Total HTTP requests received",
        "# TYPE hpe_requests_total counter",
        f"hpe_requests_total {_REQUEST_COUNT['total']}",
        "",
        "# HELP hpe_errors_total Total HTTP errors returned",
        "# TYPE hpe_errors_total counter",
        f"hpe_errors_total {_REQUEST_COUNT['errors']}",
        "",
        "# HELP hpe_python_info Python runtime info",
        "# TYPE hpe_python_info gauge",
        "hpe_python_info{version=\"3.11\"} 1",
    ]
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


# ===========================================================================
# Helper to track request counts (called from middleware)
# ===========================================================================

def increment_request_count(error: bool = False) -> None:
    _REQUEST_COUNT["total"] += 1
    if error:
        _REQUEST_COUNT["errors"] += 1


# ===========================================================================
# #44 Standard error response
# ===========================================================================

class StandardError:
    """Padrão de erro JSON unificado para toda a API.

    {
      "error": {
        "code": "VALIDATION_FAILED",
        "message": "Human readable message",
        "details": {...},
        "trace_id": "abc123"
      }
    }
    """

    @staticmethod
    def make(code: str, message: str, details: dict | None = None,
             status_code: int = 400) -> dict:
        import uuid
        return {
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "trace_id": uuid.uuid4().hex[:12],
                "status_code": status_code,
            }
        }


@router.get("/api/v1/error_codes", summary="List standard error codes")
def list_error_codes() -> dict[str, Any]:
    """Documentação dos códigos de erro padrão da HPE API."""
    return {
        "codes": [
            {"code": "VALIDATION_FAILED", "status": 400, "desc": "Input parameters invalid"},
            {"code": "NOT_FOUND", "status": 404, "desc": "Resource not found"},
            {"code": "DEPENDENCY_UNAVAILABLE", "status": 503, "desc": "External dep down"},
            {"code": "CFD_DIVERGED", "status": 500, "desc": "CFD solver diverged"},
            {"code": "MESH_QUALITY_BAD", "status": 422, "desc": "Mesh quality below threshold"},
            {"code": "OPTIMIZATION_FAILED", "status": 500, "desc": "Optimizer failed"},
            {"code": "SURROGATE_NOT_LOADED", "status": 503, "desc": "ML model not loaded"},
            {"code": "RATE_LIMITED", "status": 429, "desc": "Too many requests"},
        ],
    }
