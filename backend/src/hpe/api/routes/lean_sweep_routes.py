"""Lean / Sweep / Bow analysis routes.

POST /api/v1/analysis/lean_sweep
    Runs sizing + geometry, then computes stacking-law metrics:
    lean angles (hub/mid/shroud), sweep angle, bow fraction,
    LE/TE meridional lines and stacking line.
"""

from __future__ import annotations

import math
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["analysis"])


class LeanSweepRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Flow rate Q [m\u00b3/s]")
    head: float = Field(..., gt=0, description="Head H [m]")
    rpm: float = Field(..., gt=0, description="Rotational speed [RPM]")
    lean_angle: float = Field(0.0, description="Applied lean angle [deg]")
    sweep_angle: float = Field(0.0, description="Applied sweep angle [deg]")
    n_span_points: int = Field(16, ge=4, le=64)
    n_blade_points: int = Field(60, ge=20, le=200)


class RZPoint(BaseModel):
    r: float
    z: float


class RThetaPoint(BaseModel):
    r: float
    theta: float


class LeanSweepResponse(BaseModel):
    lean_angles: List[float]  # [hub, mid, shroud] degrees
    sweep_angle: float        # degrees
    bow_fraction: float       # 0-1
    le_line: List[RZPoint]    # [{r, z}] from hub to shroud
    te_line: List[RZPoint]    # [{r, z}] from hub to shroud
    stacking_line: List[RThetaPoint]  # [{r, theta}] at each span
    recommendations: List[str]


@router.post("/analysis/lean_sweep", response_model=LeanSweepResponse)
def lean_sweep_analysis(req: LeanSweepRequest) -> LeanSweepResponse:
    """Compute lean, sweep and bow metrics for the impeller blade stacking."""
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
    sizing = run_sizing(op)

    mp = sizing.meridional_profile
    d1 = sizing.impeller_d1
    d2 = sizing.impeller_d2
    b2 = sizing.impeller_b2
    b1 = float(mp.get("b1", b2 * 1.2))
    r1 = d1 / 2.0
    r1_hub = float(mp.get("d1_hub", d1 * 0.35)) / 2.0
    r2 = d2 / 2.0

    n_chord = req.n_blade_points
    n_span = req.n_span_points

    beta1_rad = math.radians(sizing.beta1)
    beta2_rad = math.radians(sizing.beta2)
    lean_rad = math.radians(req.lean_angle)
    sweep_m = math.tan(math.radians(req.sweep_angle)) * b2 if req.sweep_angle != 0 else 0.0

    # Build meridional curves
    z_axial = 0.55 * (r2 - r1)
    hub_rz: list[tuple[float, float]] = []
    shroud_rz: list[tuple[float, float]] = []
    for i in range(n_chord):
        t = i / (n_chord - 1)
        arc = math.pi / 2 * t
        sin_a = math.sin(arc)
        cos_a = math.cos(arc)
        r_h = r1_hub + (r2 - r1_hub) * sin_a
        z_h = z_axial * (1.0 - sin_a)
        b_t = b1 + t * (b2 - b1)
        r_s = r_h + b_t * cos_a
        z_s = z_h + b_t * sin_a
        hub_rz.append((r_h, z_h))
        shroud_rz.append((r_s, z_s))

    # Build camber lines at each span station
    camber_lines: list[list[tuple[float, float, float]]] = []  # [span][chord] -> (r, z, theta)
    for k in range(n_span):
        s = k / max(1, n_span - 1)
        theta = 0.0
        camber: list[tuple[float, float, float]] = []
        for i in range(n_chord):
            t = i / (n_chord - 1)
            r = hub_rz[i][0] + s * (shroud_rz[i][0] - hub_rz[i][0])
            z = hub_rz[i][1] + s * (shroud_rz[i][1] - hub_rz[i][1])
            beta = beta1_rad + t * (beta2_rad - beta1_rad)
            # Apply stacking offsets
            lean_offset = lean_rad * s
            z_offset = sweep_m * s
            camber.append((r, z + z_offset, theta + lean_offset))
            if i < n_chord - 1:
                r_next = hub_rz[i + 1][0] + s * (shroud_rz[i + 1][0] - hub_rz[i + 1][0])
                beta_next = beta1_rad + (i + 1) / (n_chord - 1) * (beta2_rad - beta1_rad)
                b_mid = (beta + beta_next) / 2
                r_mid = (r + r_next) / 2
                dr = r_next - r
                if abs(math.tan(b_mid)) > 1e-10 and r_mid > 1e-6:
                    theta += dr / (r_mid * math.tan(b_mid))
        camber_lines.append(camber)

    # Extract LE line (first chord point at each span)
    le_line = [RZPoint(r=round(cl[0][0] * 1000, 3), z=round(cl[0][1] * 1000, 3))
               for cl in camber_lines]
    # Extract TE line (last chord point at each span)
    te_line = [RZPoint(r=round(cl[-1][0] * 1000, 3), z=round(cl[-1][1] * 1000, 3))
               for cl in camber_lines]

    # Stacking line: mid-chord theta at each span
    mid_chord_idx = n_chord // 2
    stacking_line = [
        RThetaPoint(
            r=round(cl[mid_chord_idx][0] * 1000, 3),
            theta=round(math.degrees(cl[mid_chord_idx][2]), 3),
        )
        for cl in camber_lines
    ]

    # Compute lean angles at hub, mid, shroud
    # Lean = angle between stacking axis and radial direction in r-theta plane
    lean_angles: list[float] = []
    for idx in [0, n_span // 2, n_span - 1]:
        if idx < len(stacking_line):
            lean_angles.append(round(stacking_line[idx].theta, 2))
        else:
            lean_angles.append(0.0)

    # Sweep angle: angle of LE line vs radial in r-z plane
    if len(le_line) >= 2:
        dr_le = le_line[-1].r - le_line[0].r
        dz_le = le_line[-1].z - le_line[0].z
        sweep_deg = round(math.degrees(math.atan2(dz_le, dr_le + 1e-12)), 2)
    else:
        sweep_deg = 0.0

    # Bow fraction: max deviation of stacking line from straight hub-to-shroud
    if len(stacking_line) >= 3:
        t0 = stacking_line[0].theta
        t1 = stacking_line[-1].theta
        span_range = t1 - t0
        max_dev = 0.0
        for k in range(1, len(stacking_line) - 1):
            s_frac = k / (len(stacking_line) - 1)
            expected = t0 + s_frac * span_range
            dev = abs(stacking_line[k].theta - expected)
            max_dev = max(max_dev, dev)
        bow_frac = round(max_dev / (abs(span_range) + 1e-10), 4)
    else:
        bow_frac = 0.0

    # Recommendations
    recommendations: list[str] = []
    if abs(lean_angles[1]) < 2.0:
        recommendations.append("Lean neutro: boa distribuicao de carga uniforme hub-to-shroud.")
    elif lean_angles[1] > 5.0:
        recommendations.append("Lean positivo alto: tende a descarregar o shroud, aumentando eficiencia em high-Nq.")
    elif lean_angles[1] < -5.0:
        recommendations.append("Lean negativo: carrega mais o shroud, pode piorar cavitacao.")

    if abs(sweep_deg) < 5.0:
        recommendations.append("Sweep minimo: LE essencialmente radial.")
    elif sweep_deg > 15.0:
        recommendations.append("Forward sweep significativo: pode melhorar NPSHr mas aumenta stress no LE.")
    elif sweep_deg < -15.0:
        recommendations.append("Backward sweep: melhora estabilidade mas pode aumentar perdas de incidencia.")

    if bow_frac < 0.05:
        recommendations.append("Stacking praticamente linear (bow < 5%).")
    elif bow_frac > 0.15:
        recommendations.append("Bow pronunciado (>15%): verificar distribuicao de tensoes e fadiga.")

    return LeanSweepResponse(
        lean_angles=lean_angles,
        sweep_angle=sweep_deg,
        bow_fraction=bow_frac,
        le_line=le_line,
        te_line=te_line,
        stacking_line=stacking_line,
        recommendations=recommendations,
    )
