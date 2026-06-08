"""Noise prediction API routes.

Endpoints:
    POST /api/v1/analysis/noise — full noise prediction for a pump design.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["noise"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class NoiseRequest(BaseModel):
    """Input for noise prediction."""

    flow_rate: float = Field(..., gt=0, description="Volume flow rate Q [m³/s]")
    head: float = Field(..., gt=0, description="Total head H [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [rev/min]")
    fluid: str = Field("water", description="Working fluid name")
    rho: float = Field(998.2, gt=0, description="Fluid density [kg/m³]")
    c_sound: float = Field(1480.0, gt=0, description="Speed of sound [m/s]")
    noise_limit_dB: Optional[float] = Field(
        None, description="Optional noise limit [dB] for pass/fail check"
    )


class NoiseHarmonic(BaseModel):
    order: float
    freq_hz: float
    lw_dB: float


class NoiseSpectrumPoint(BaseModel):
    frequency_hz: float
    lw_dB: float
    lw_A_dB: float


class NoiseResponse(BaseModel):
    """Noise prediction output."""

    lw_total_dB: float
    lw_A_weighted_dB: float
    bpf_hz: float
    dominant_source: str
    lw_broadband_dB: float
    lw_tonal_dB: float
    lw_cavitation_dB: float
    cavitation_onset: bool
    harmonics: List[NoiseHarmonic]
    spectrum: List[NoiseSpectrumPoint]
    above_limit: Optional[bool] = None
    warnings: List[str]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analysis/noise", response_model=NoiseResponse)
def predict_noise(req: NoiseRequest) -> NoiseResponse:
    """Run full noise prediction for a centrifugal pump design.

    Performs 1D sizing to obtain impeller geometry, then evaluates
    broadband, tonal (BPF), and cavitation noise sources.

    Args:
        req: Noise prediction request with operating point and fluid data.

    Returns:
        NoiseResponse with total levels, spectrum, and warnings.
    """
    from hpe.core.models import OperatingPoint
    from hpe.physics.noise_prediction import NoisePredictor
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    try:
        sizing = run_sizing(op)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    predictor = NoisePredictor(rho=req.rho, c_sound=req.c_sound)
    result = predictor.predict(sizing, rpm=req.rpm, fluid=req.fluid)

    above_limit: Optional[bool] = None
    if req.noise_limit_dB is not None:
        above_limit = result.is_above_limit(req.noise_limit_dB)

    return NoiseResponse(
        lw_total_dB=result.lw_total_dB,
        lw_A_weighted_dB=result.lw_A_weighted_dB,
        bpf_hz=result.bpf_hz,
        dominant_source=result.dominant_source,
        lw_broadband_dB=result.lw_broadband_dB,
        lw_tonal_dB=result.lw_tonal_dB,
        lw_cavitation_dB=result.lw_cavitation_dB,
        cavitation_onset=result.cavitation_onset,
        harmonics=[
            NoiseHarmonic(order=h["order"], freq_hz=h["freq_hz"], lw_dB=h["lw_dB"])
            for h in result.harmonics
        ],
        spectrum=[
            NoiseSpectrumPoint(
                frequency_hz=s["frequency_hz"], lw_dB=s["lw_dB"], lw_A_dB=s["lw_A_dB"]
            )
            for s in result.spectrum
        ],
        above_limit=above_limit,
        warnings=result.warnings,
    )
