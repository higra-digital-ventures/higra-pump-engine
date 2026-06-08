"""Volute sizing endpoint — expõe /volute/run na API.

Standalone FastAPI router.  Mount in your app with:

    from hpe.api.volute_endpoint import router as volute_router
    app.include_router(volute_router)

Endpoints
---------
    POST /volute/run   — Size a spiral volute from operating-point inputs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(tags=["Volute"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class VoluteInput(BaseModel):
    """Operating point for volute sizing."""

    Q: float = Field(..., gt=0, description="Flow rate [m³/s]")
    H: float = Field(..., gt=0, description="Total head per stage [m]")
    n: float = Field(..., gt=0, description="Rotational speed [rpm]")
    tongue_clearance: float = Field(
        1.05, ge=1.0, le=1.3,
        description="Tongue radius ratio r_tongue/r2 (default 1.05)",
    )
    velocity_ratio: float = Field(
        0.9, gt=0.3, le=1.5,
        description="u3/u2 velocity ratio at volute exit (default 0.9)",
    )


class VoluteOutput(BaseModel):
    """Volute sizing results in mm and mm²."""

    # Primary dimensions (mm / mm²)
    throat_area_mm2: float = Field(..., description="Discharge throat area [mm²]")
    tongue_radius_mm: float = Field(..., description="Cutwater radius from shaft centre [mm]")
    exit_diameter_mm: float = Field(..., description="Discharge pipe inner diameter [mm]")
    casing_width_mm: float = Field(..., description="Volute casing width at discharge [mm]")
    spiral_length_mm: float = Field(..., description="Approximate spiral centreline length [mm]")

    # Reference impeller dimensions
    D2_mm: float = Field(..., description="Impeller tip diameter [mm]")
    b2_mm: float = Field(..., description="Impeller outlet width [mm]")
    r3_mm: float = Field(..., description="Volute base (inlet) radius [mm]")

    # Sizing parameters
    ns: float = Field(..., description="Specific speed Ns [rpm, m³/s, m]")
    nq: float = Field(..., description="European specific speed Nq = Ns/51.65")
    cu2: float = Field(..., description="Tangential velocity at impeller outlet cu2 [m/s]")
    q_m3s: float = Field(..., description="Internal flow rate used [m³/s]")

    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/volute/run", response_model=VoluteOutput)
async def run_volute(inp: VoluteInput) -> VoluteOutput:
    """Size a spiral volute from an operating point.

    Internally:
      1. Runs the 1D meanline sizing (hpe.sizing.meanline.run_sizing)
      2. Passes SizingResult to the volute pipeline (hpe.geometry.volute.pipeline)
      3. Returns all key dimensions converted to mm / mm².

    Parameters
    ----------
    inp : VoluteInput
        Q (m³/s), H (m), n (rpm), tongue_clearance, velocity_ratio.

    Returns
    -------
    VoluteOutput
        Throat area, tongue radius, exit diameter, casing width, spiral
        length, reference impeller dimensions, and any design warnings.
    """
    try:
        from hpe.core.models import OperatingPoint
        from hpe.sizing.meanline import run_sizing
        from hpe.geometry.volute.pipeline import run_volute_pipeline

        # 1. Run 1D sizing
        op = OperatingPoint(
            flow_rate=inp.Q,
            head=inp.H,
            rpm=inp.n,
        )
        sr = run_sizing(op)

        # 2. Run volute pipeline
        vr = run_volute_pipeline(
            sr,
            tongue_clearance=inp.tongue_clearance,
            velocity_ratio=inp.velocity_ratio,
        )

        # 3. Extract cu2 and q for reference (already used inside pipeline)
        from hpe.geometry.volute.pipeline import _extract_cu2, _extract_flow_rate
        cu2 = _extract_cu2(sr)
        q_m3s = _extract_flow_rate(sr)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("run_volute: unexpected error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    # Convert to mm / mm²
    M2_TO_MM2 = 1e6
    M_TO_MM = 1e3

    return VoluteOutput(
        throat_area_mm2=round(vr.throat_area_m2 * M2_TO_MM2, 2),
        tongue_radius_mm=round(vr.tongue_radius_m * M_TO_MM, 2),
        exit_diameter_mm=round(vr.exit_diameter_m * M_TO_MM, 2),
        casing_width_mm=round(vr.casing_width_m * M_TO_MM, 2),
        spiral_length_mm=round(vr.spiral_length_m * M_TO_MM, 2),
        D2_mm=round(sr.impeller_d2 * M_TO_MM, 2),
        b2_mm=round(sr.impeller_b2 * M_TO_MM, 2),
        r3_mm=round(vr.sizing.r3 * M_TO_MM, 2),
        ns=round(sr.specific_speed_ns, 3),
        nq=round(sr.specific_speed_nq, 4),
        cu2=round(cu2, 3),
        q_m3s=round(q_m3s, 6),
        warnings=vr.warnings,
    )
