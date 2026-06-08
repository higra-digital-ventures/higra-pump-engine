"""Surrogate model API endpoints (#30).

Endpoints:
    POST /api/v1/surrogate/train   — train EtaSurrogate on synthetic data
    POST /api/v1/surrogate/predict — predict η for an impeller design
    GET  /api/v1/surrogate/status  — model status / last metrics
"""

from __future__ import annotations

import functools
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["surrogate"])

# ── Module-level singleton ─────────────────────────────────────────────────────
# The surrogate is loaded once and cached in memory.  Training replaces it.

@functools.lru_cache(maxsize=1)
def _get_surrogate_cached():
    """Return the global EtaSurrogate, loading from disk if available."""
    from hpe.ai.surrogate.eta_predictor import EtaSurrogate
    try:
        return EtaSurrogate.load()
    except FileNotFoundError:
        return EtaSurrogate()


def _surrogate():
    return _get_surrogate_cached()


# ── Schemas ───────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    n_samples: int = Field(600, ge=100, le=5000,
                           description="Number of synthetic training samples")
    flow_rate: float = Field(0.05, gt=0, description="Reference Q [m³/s]")
    head: float = Field(30.0, gt=0, description="Reference H [m]")
    rpm: float = Field(1750.0, gt=0, description="Reference N [rpm]")
    seed: int = Field(42)


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    r2_train: float
    r2_cv: float
    mae_cv: float
    n_train: int
    model_path: str


class PredictRequest(BaseModel):
    nq: float = Field(..., gt=0, description="Specific speed [—]")
    d2: float = Field(..., gt=0, description="Outlet diameter [m]")
    d1: float = Field(..., gt=0, description="Inlet diameter [m]")
    b2: float = Field(..., gt=0, description="Outlet width [m]")
    z: int = Field(..., ge=3, le=20, description="Blade count")
    beta2: float = Field(..., gt=0, lt=90, description="Outlet blade angle [deg]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rpm]")


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eta_predicted: float
    eta_pct: float
    model_trained: bool
    r2_cv: float | None = None


class StatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    trained: bool
    metrics: dict[str, float]
    model_path: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/surrogate/train", response_model=TrainResponse)
def train_surrogate(req: TrainRequest) -> TrainResponse:
    """Train the η surrogate model on physics-generated synthetic data.

    Clears the cached singleton so the newly trained model is used.
    """
    from hpe.ai.surrogate.eta_predictor import EtaSurrogate

    surr = EtaSurrogate()
    try:
        metrics = surr.train(
            n_samples=req.n_samples,
            flow_rate=req.flow_rate,
            head=req.head,
            rpm=req.rpm,
            seed=req.seed,
            save=True,
        )
    except Exception as exc:
        log.exception("Surrogate training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Replace cached singleton
    _get_surrogate_cached.cache_clear()

    return TrainResponse(
        r2_train=metrics["r2_train"],
        r2_cv=metrics["r2_cv"],
        mae_cv=metrics["mae_cv"],
        n_train=metrics["n_train"],
        model_path=str(surr.model_path),
    )


@router.post("/surrogate/predict", response_model=PredictResponse)
def predict_surrogate(req: PredictRequest) -> PredictResponse:
    """Predict hydraulic efficiency η for a given impeller design.

    If no model is trained, falls back to physics model for a single point.
    """
    surr = _surrogate()

    if not surr.is_trained:
        # Fallback: run physics model
        from hpe.core.models import OperatingPoint
        from hpe.sizing import run_sizing
        try:
            # Rough Q estimate from Nq = n·√Q / H^0.75
            q_approx = (req.nq * (req.rpm ** -1) * req.head ** 0.75) ** 2
            r = run_sizing(OperatingPoint(
                flow_rate=max(1e-4, q_approx),
                head=req.nq ** (4/3) / (req.rpm ** (2/3)) * 9.81,  # rough
                rpm=req.rpm,
            ))
            eta = r.estimated_efficiency
        except Exception:
            eta = 0.0

        return PredictResponse(
            eta_predicted=eta,
            eta_pct=eta * 100,
            model_trained=False,
            r2_cv=None,
        )

    try:
        eta = surr.predict(
            nq=req.nq, d2=req.d2, d1=req.d1, b2=req.b2,
            z=req.z, beta2=req.beta2, rpm=req.rpm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        eta_predicted=eta,
        eta_pct=round(eta * 100, 2),
        model_trained=True,
        r2_cv=surr.metrics.get("r2_cv"),
    )


@router.get("/surrogate/status", response_model=StatusResponse)
def surrogate_status() -> StatusResponse:
    """Return current surrogate model status and training metrics."""
    surr = _surrogate()
    return StatusResponse(
        trained=surr.is_trained,
        metrics=surr.metrics,
        model_path=str(surr.model_path),
    )
