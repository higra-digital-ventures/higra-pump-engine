"""WebSocket endpoint and REST trigger for HPE Pipeline runs.

Exposes two routes via an ``APIRouter`` that can be mounted in any FastAPI
app **without modifying** the existing ``hpe.api.main`` or
``hpe.api.app`` modules:

  POST /pipeline/run
      Start a pipeline run.  Returns ``run_id`` for status polling.
      Falls back to synchronous execution when Celery is unavailable.

  WS   /ws/pipeline/{run_id}
      Stream ``PipelineStatus`` JSON every 500 ms until done or failed.
      Closes the connection automatically when the run finishes.

Integration example (in hpe.api.app or another FastAPI entrypoint)
------------------------------------------------------------------
    from hpe.api.websocket import router as pipeline_router
    app.include_router(pipeline_router)

WebSocket protocol
------------------
Client → Server (optional)::

    {"action": "subscribe"}

Server → Client (every 500 ms)::

    {
      "run_id": "...",
      "status": "running",
      "stage": "geometry",
      "progress": 35,
      "elapsed_s": 1.2,
      "eta_s": 2.3,
      "message": "Building CAD"
    }

Server closes connection when ``status`` is ``"completed"`` or ``"failed"``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["Pipeline"])

# Maximum polling iterations before timeout (7200 × 0.5s = 3600s / 1h)
_MAX_ITERATIONS = 7200
_POLL_INTERVAL = 0.5  # seconds


# ---------------------------------------------------------------------------
# WebSocket status stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/pipeline/{run_id}")
async def pipeline_status_ws(websocket: WebSocket, run_id: str) -> None:
    """Stream real-time pipeline status updates via WebSocket.

    Parameters
    ----------
    run_id : str
        The run identifier returned by ``POST /pipeline/run``.
    """
    await websocket.accept()
    log.debug("WS /ws/pipeline/%s: client connected", run_id)

    try:
        # Optional initial message from client (e.g. {"action": "subscribe"})
        # We don't block on it — just consume if available within 1s.
        try:
            _ = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass  # No initial message — proceed anyway

        from hpe.orchestrator.status import tracker

        for _ in range(_MAX_ITERATIONS):
            status = tracker.get(run_id)

            if status is not None:
                await websocket.send_json(status.to_dict())
                if status.status in ("completed", "failed"):
                    log.debug("WS /ws/pipeline/%s: run finished (%s)", run_id, status.status)
                    break
            else:
                await websocket.send_json({
                    "run_id": run_id,
                    "status": "not_found",
                    "message": "No status found for this run_id",
                })

            await asyncio.sleep(_POLL_INTERVAL)

    except WebSocketDisconnect:
        log.debug("WS /ws/pipeline/%s: client disconnected", run_id)
    except Exception as exc:
        log.exception("WS /ws/pipeline/%s: unexpected error — %s", run_id, exc)
        try:
            await websocket.send_json({"run_id": run_id, "status": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST trigger: POST /pipeline/run
# ---------------------------------------------------------------------------

@router.post("/pipeline/run", summary="Start async pipeline run")
async def start_pipeline(op: dict) -> dict:
    """Start a full HPE design pipeline run.

    Accepts an ``OperatingPoint`` dict and returns a ``run_id`` that can be
    used to poll status via ``GET /ws/pipeline/{run_id}``.

    Body keys (at least one of the following sets):

    * ``Q`` (m³/h), ``H`` (m), ``n`` (rpm)
    * ``flow_rate`` (m³/s), ``head`` (m), ``rpm`` (rpm)

    Optional: ``run_cfd`` (bool, default false), ``project_id``, ``notes``,
    ``tags`` (list[str]).

    Returns
    -------
    dict
        If Celery is available:
            ``{"run_id": str, "task_id": str, "mode": "async"}``
        Fallback synchronous:
            ``{"run_id": str, "mode": "sync", "D2_mm": float, "eta": float,
               "elapsed_ms": float}``
    """
    run_id = str(uuid.uuid4())
    run_cfd: bool = bool(op.pop("run_cfd", False))

    # Inject run_id so tasks can update status tracker
    op_with_id = {**op, "_run_id": run_id}

    from hpe.orchestrator.status import tracker, PipelineStatus

    # Register pending status immediately so WebSocket clients can connect
    tracker.set(run_id, PipelineStatus.pending(run_id))

    # ── Try Celery (async) ─────────────────────────────────────────────
    try:
        from hpe.orchestrator.tasks import run_full_pipeline_task  # type: ignore[attr-defined]

        if not callable(getattr(run_full_pipeline_task, "delay", None)):
            raise ImportError("Celery not available")

        # Verify Celery is reachable (ping broker)
        from hpe.orchestrator.config import celery_app as _celery_app
        if _celery_app is None:
            raise ImportError("celery_app is None")

        task = run_full_pipeline_task.delay(op_with_id, run_cfd)
        log.info("start_pipeline: async run_id=%s task_id=%s", run_id, task.id)
        return {"run_id": run_id, "task_id": task.id, "mode": "async"}

    except Exception as celery_exc:
        log.info(
            "start_pipeline: Celery unavailable (%s) — falling back to sync execution",
            celery_exc,
        )

    # ── Synchronous fallback ───────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = await _run_sync(op_with_id, run_id, run_cfd)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        result["elapsed_ms"] = elapsed_ms
        result["mode"] = "sync"
        result["run_id"] = run_id
        return result
    except Exception as exc:
        tracker.fail(run_id, str(exc))
        log.exception("start_pipeline: sync fallback failed — %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "run_id": run_id},
        )


async def _run_sync(op_with_id: dict, run_id: str, run_cfd: bool) -> dict:
    """Execute the pipeline synchronously in an executor thread."""
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor

    def _execute() -> dict:
        from hpe.orchestrator.tasks import run_full_pipeline_task
        return run_full_pipeline_task(op_with_id, run_cfd)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hpe-sync") as ex:
        result: dict = await loop.run_in_executor(ex, _execute)

    return result


# ---------------------------------------------------------------------------
# WebSocket: GET /ws/cfd/{run_id}/residuals — stream OpenFOAM convergence
# ---------------------------------------------------------------------------

@router.websocket("/ws/cfd/{run_id}/residuals")
async def cfd_residuals_ws(websocket: WebSocket, run_id: str) -> None:
    """Stream OpenFOAM convergence residuals in real time.

    Tails ``log.MRFSimpleFoam`` (or ``log.simpleFoam``) in the case
    directory associated with *run_id* and emits parsed residuals at
    500 ms intervals until the solver stops or the client disconnects.

    Message format (server → client)::

        {
          "run_id": "abc123",
          "iteration": 42,
          "residuals": {"Ux": 1.2e-4, "p": 8.5e-5, ...},
          "converged_fields": ["p"],
          "should_stop": false,
          "reason": "running",
          "message": ""
        }
    """
    await websocket.accept()
    log.debug("WS /ws/cfd/%s/residuals: client connected", run_id)

    try:
        # Locate case directory via the shared run registry
        from hpe.api.routes.cfd_loop_routes import _runs
        entry = _runs.get(run_id)

        if entry is None:
            await websocket.send_json({
                "run_id": run_id, "status": "not_found",
                "error": f"No CFD run found for run_id={run_id}",
            })
            return

        work_dir: Path = entry.get("work_dir", Path("cfd_runs") / run_id)

        from hpe.cfd.openfoam.convergence import ConvergenceMonitor, ConvergenceCriteria

        criteria = ConvergenceCriteria(tol=1e-4, window=20, divergence_factor=100.0)
        monitor = ConvergenceMonitor(case_dir=work_dir, criteria=criteria)

        for _ in range(_MAX_ITERATIONS):
            status = monitor.update()

            payload: dict = {
                "run_id":          run_id,
                "iteration":       status.iteration,
                "residuals":       {k: round(v, 8) for k, v in (status.residuals or {}).items()},
                "converged_fields": list(status.converged_fields or []),
                "should_stop":     status.should_stop,
                "reason":          status.reason.value if status.reason else "running",
                "message":         status.message or "",
            }
            await websocket.send_json(payload)

            if status.should_stop:
                log.debug("WS /ws/cfd/%s/residuals: solver stopped (%s)", run_id, status.reason)
                break

            # Also stop if the run was cancelled or completed via API
            if entry.get("status") in ("completed", "cancelled", "failed"):
                break

            await asyncio.sleep(_POLL_INTERVAL)

    except WebSocketDisconnect:
        log.debug("WS /ws/cfd/%s/residuals: client disconnected", run_id)
    except Exception as exc:
        log.exception("WS /ws/cfd/%s/residuals: error — %s", run_id, exc)
        try:
            await websocket.send_json({"run_id": run_id, "status": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST: GET /pipeline/status/{run_id}  — REST alternative to WebSocket
# ---------------------------------------------------------------------------

@router.get("/pipeline/status/{run_id}", summary="Poll pipeline status (REST)")
async def get_pipeline_status(run_id: str) -> dict:
    """Return the current status of a pipeline run (REST polling alternative).

    Returns
    -------
    dict
        ``PipelineStatus`` dict, or ``{"run_id": ..., "status": "not_found"}``
    """
    from hpe.orchestrator.status import tracker

    status = tracker.get(run_id)
    if status is None:
        return {"run_id": run_id, "status": "not_found"}
    return status.to_dict()


# ---------------------------------------------------------------------------
# REST: GET /pipeline/versions — list saved design versions
# ---------------------------------------------------------------------------

@router.get("/pipeline/versions", summary="List saved design versions")
async def list_versions(
    project_id: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """Return the most recent design versions from local storage.

    Parameters
    ----------
    project_id : str, optional
        Filter by project.
    limit : int
        Max records (default 20, max 200).
    """
    limit = min(max(1, limit), 200)
    try:
        from hpe.orchestrator.versions import load_versions

        versions = load_versions(project_id=project_id, limit=limit)
        return {
            "total": len(versions),
            "versions": [v.to_dict() for v in versions],
        }
    except Exception as exc:
        log.warning("list_versions: failed — %s", exc)
        return {"total": 0, "versions": [], "error": str(exc)}
