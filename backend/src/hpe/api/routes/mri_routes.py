"""Meridional (MRI) generator API routes.

Endpoints:
    POST /api/v1/meridional/generate             — generate from parameters
    POST /api/v1/meridional/from_sizing           — auto-generate from sizing result
    POST /api/v1/meridional/from_control_points   — interpolate from drag-editor control points
    GET  /api/v1/meridional/templates             — list available templates
    POST /api/v1/meridional/validate              — validate geometry
"""

from __future__ import annotations

from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hpe.geometry.meridional.mri_generator import (
    MRIGenerator,
    MRIParams,
    MRITemplate,
)

router = APIRouter(prefix="/api/v1/meridional", tags=["meridional"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class MeridionalGenerateRequest(BaseModel):
    """Parameters for meridional profile generation."""

    inlet_radius: float = Field(..., gt=0, description="Inlet (shroud tip) radius [m]")
    outlet_radius: float = Field(..., gt=0, description="Outlet (impeller exit) radius [m]")
    axial_length: float = Field(..., gt=0, description="Total axial extent [m]")
    hub_curvature: float = Field(0.5, ge=0, le=1)
    shroud_curvature: float = Field(0.5, ge=0, le=1)
    inlet_angle_hub: float = Field(90.0, ge=0, le=180)
    inlet_angle_shroud: float = Field(90.0, ge=0, le=180)
    outlet_angle_hub: float = Field(90.0, ge=0, le=180)
    outlet_angle_shroud: float = Field(90.0, ge=0, le=180)
    hub_inlet_radius: float = Field(0.0, ge=0, description="Hub r at inlet (shaft). 0 = auto")
    passage_width_inlet: float = Field(0.0, ge=0, description="b1 at inlet. 0 = auto")
    n_points: int = Field(50, ge=10, le=500)


class MeridionalFromSizingRequest(BaseModel):
    """Auto-generate meridional from sizing dimensions."""

    impeller_d1: float = Field(0.0, ge=0, description="Inlet diameter [m]. 0 = auto")
    impeller_d2: float = Field(..., gt=0, description="Outlet diameter [m]")
    impeller_b1: float = Field(0.0, ge=0, description="Inlet width [m]. 0 = auto")
    impeller_b2: float = Field(..., gt=0, description="Outlet width [m]")
    shaft_diameter: float = Field(0.0, ge=0, description="Shaft diameter [m]. 0 = auto")
    n_points: int = Field(50, ge=10, le=500)


class MeridionalValidateRequest(MeridionalGenerateRequest):
    """Same params as generate, returns validation diagnostics."""
    pass


class ProfileResponse(BaseModel):
    hub_rz: List[List[float]]
    shroud_rz: List[List[float]]
    n_points: int


class TemplateInfo(BaseModel):
    name: str
    description: str


class TemplatesResponse(BaseModel):
    templates: List[TemplateInfo]


class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    min_passage_width: float
    max_curvature_hub: float
    max_curvature_shroud: float


class RZPoint(BaseModel):
    r: float
    z: float


class ControlPointsRequest(BaseModel):
    """Control points from the interactive drag editor."""

    hub_points: List[RZPoint] = Field(..., min_length=3, description="Hub Bezier control points (r, z) in metres")
    shroud_points: List[RZPoint] = Field(..., min_length=3, description="Shroud Bezier control points (r, z) in metres")
    n_output_points: int = Field(50, ge=10, le=500, description="Number of points on each interpolated curve")


class ControlPointsMetrics(BaseModel):
    passage_area_ratio: float
    l_d2: float
    curvature_radius_hub: float
    curvature_radius_shroud: float


class ControlPointsResponse(BaseModel):
    hub_rz: List[List[float]]
    shroud_rz: List[List[float]]
    n_points: int
    metrics: ControlPointsMetrics


class ControlPointsValidateRequest(BaseModel):
    """Validate control-point geometry from the drag editor."""

    hub_points: List[RZPoint] = Field(..., min_length=3)
    shroud_points: List[RZPoint] = Field(..., min_length=3)
    n_output_points: int = Field(50, ge=10, le=500)


class ControlPointsValidateResponse(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: ControlPointsMetrics


class TemplateControlPointsRequest(BaseModel):
    """Request preset control points for a template type."""

    template: str = Field(..., description="Template name: radial_pump, mixed_flow, francis_turbine, axial")
    d2: float = Field(0.3, gt=0, description="Reference outlet diameter [m]")
    b2: float = Field(0.02, gt=0, description="Reference outlet width [m]")


class TemplateControlPointsResponse(BaseModel):
    hub_points: List[RZPoint]
    shroud_points: List[RZPoint]


# ---------------------------------------------------------------------------
# Helper: cubic Bezier / composite Bezier interpolation
# ---------------------------------------------------------------------------

def _cubic_bezier_segment(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray,
) -> np.ndarray:
    """Evaluate cubic Bezier at parameter values *t* (0..1)."""
    u = 1.0 - t
    return (
        np.outer(u**3, p0)
        + np.outer(3 * u**2 * t, p1)
        + np.outer(3 * u * t**2, p2)
        + np.outer(t**3, p3)
    )


def _interpolate_composite_bezier(
    control_points: List[RZPoint], n_out: int,
) -> np.ndarray:
    """Interpolate a composite cubic Bezier through ordered control points.

    When exactly 4 points are given, a single cubic segment is used.
    Otherwise the points are treated as a Catmull-Rom-style chain that
    is converted to piecewise cubic Bezier segments so that the resulting
    curve passes through every control point with C1 continuity.

    Returns:
        ndarray of shape (n_out, 2) with columns (r, z).
    """
    cps = np.array([[p.r, p.z] for p in control_points])
    n_cp = len(cps)

    if n_cp < 2:
        raise ValueError("Need at least 2 control points")

    if n_cp == 2:
        # Linear interpolation
        t = np.linspace(0, 1, n_out).reshape(-1, 1)
        return cps[0] + t * (cps[1] - cps[0])

    if n_cp == 3:
        # Quadratic Bezier elevated to cubic
        p0, p1, p2 = cps
        cp0 = p0
        cp1 = p0 + 2.0 / 3.0 * (p1 - p0)
        cp2 = p2 + 2.0 / 3.0 * (p1 - p2)
        cp3 = p2
        t = np.linspace(0, 1, n_out)
        return _cubic_bezier_segment(cp0, cp1, cp2, cp3, t)

    if n_cp == 4:
        # Single cubic Bezier segment
        t = np.linspace(0, 1, n_out)
        return _cubic_bezier_segment(cps[0], cps[1], cps[2], cps[3], t)

    # General case: Catmull-Rom -> cubic Bezier segments
    n_seg = n_cp - 1
    pts_per_seg = max(2, n_out // n_seg)
    all_pts: list[np.ndarray] = []

    for i in range(n_seg):
        # Catmull-Rom tangent estimation
        if i == 0:
            m0 = cps[1] - cps[0]
        else:
            m0 = 0.5 * (cps[i + 1] - cps[i - 1])

        if i == n_seg - 1:
            m1 = cps[i + 1] - cps[i]
        else:
            m1 = 0.5 * (cps[i + 2] - cps[i])

        # Convert to cubic Bezier control points
        bp0 = cps[i]
        bp1 = cps[i] + m0 / 3.0
        bp2 = cps[i + 1] - m1 / 3.0
        bp3 = cps[i + 1]

        t = np.linspace(0, 1, pts_per_seg, endpoint=(i == n_seg - 1))
        seg = _cubic_bezier_segment(bp0, bp1, bp2, bp3, t)
        all_pts.append(seg)

    curve = np.vstack(all_pts)

    # Resample to exactly n_out points
    if len(curve) != n_out:
        arc = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
        cum = np.concatenate([[0], np.cumsum(arc)])
        cum /= cum[-1]
        t_uniform = np.linspace(0, 1, n_out)
        r_interp = np.interp(t_uniform, cum, curve[:, 0])
        z_interp = np.interp(t_uniform, cum, curve[:, 1])
        curve = np.column_stack([r_interp, z_interp])

    return curve


def _compute_curvature_radius(pts: np.ndarray) -> float:
    """Return the minimum curvature radius along a discrete curve.

    Uses a three-point finite difference for curvature kappa and returns
    1/max(kappa).  Returns inf if the curve is perfectly straight.
    """
    if len(pts) < 3:
        return float("inf")

    dr = np.gradient(pts[:, 0])
    dz = np.gradient(pts[:, 1])
    d2r = np.gradient(dr)
    d2z = np.gradient(dz)

    num = np.abs(dr * d2z - dz * d2r)
    den = (dr**2 + dz**2) ** 1.5
    kappa = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    max_k = np.max(kappa)
    return 1.0 / max_k if max_k > 1e-12 else float("inf")


def _compute_metrics(
    hub: np.ndarray, shroud: np.ndarray,
) -> ControlPointsMetrics:
    """Compute passage metrics from interpolated hub/shroud arrays."""
    # Passage width at inlet (first points) and outlet (last points)
    w_inlet = np.linalg.norm(shroud[0] - hub[0])
    w_outlet = np.linalg.norm(shroud[-1] - hub[-1])
    passage_area_ratio = w_inlet / max(w_outlet, 1e-12)

    # Axial length / D2
    r_outlet_max = max(hub[-1, 0], shroud[-1, 0])
    d2 = 2.0 * r_outlet_max
    z_min = min(hub[:, 1].min(), shroud[:, 1].min())
    z_max = max(hub[:, 1].max(), shroud[:, 1].max())
    axial_length = z_max - z_min
    l_d2 = axial_length / max(d2, 1e-12)

    curv_hub = _compute_curvature_radius(hub)
    curv_shr = _compute_curvature_radius(shroud)

    return ControlPointsMetrics(
        passage_area_ratio=round(passage_area_ratio, 4),
        l_d2=round(l_d2, 4),
        curvature_radius_hub=round(curv_hub, 6),
        curvature_radius_shroud=round(curv_shr, 6),
    )


def _validate_control_points(
    hub: np.ndarray, shroud: np.ndarray,
) -> tuple[bool, list[str], list[str]]:
    """Run validation checks on interpolated curves.

    Returns (valid, errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check passage width (shroud.r should be >= hub.r at corresponding z)
    n = min(len(hub), len(shroud))
    for i in range(n):
        diff = shroud[i, 0] - hub[i, 0]
        if diff < -1e-6:
            errors.append(
                f"Hub crosses shroud at point {i}: hub_r={hub[i, 0]:.4f} > shroud_r={shroud[i, 0]:.4f}"
            )
            break

    # Check minimum passage width
    widths = np.sqrt(np.sum((shroud[:n] - hub[:n]) ** 2, axis=1))
    min_w = float(widths.min())
    if min_w < 1e-4:
        errors.append(f"Passage width too small: {min_w*1000:.2f} mm")
    elif min_w < 2e-3:
        warnings.append(f"Passage width very narrow: {min_w*1000:.1f} mm")

    # Check for self-intersecting curves (consecutive points going backwards in z)
    for name, curve in [("Hub", hub), ("Shroud", shroud)]:
        dz = np.diff(curve[:, 1])
        if np.any(dz > 0) and np.any(dz < 0):
            # Allow non-monotonic z only if it's intentional (S-curves)
            sign_changes = np.sum(np.abs(np.diff(np.sign(dz))) > 0)
            if sign_changes > 2:
                warnings.append(f"{name} curve has {sign_changes} direction changes in z")

    valid = len(errors) == 0
    return valid, errors, warnings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=ProfileResponse)
def generate_meridional(req: MeridionalGenerateRequest) -> ProfileResponse:
    """Generate meridional hub/shroud curves from parameters."""
    try:
        params = MRIParams(
            inlet_radius=req.inlet_radius,
            outlet_radius=req.outlet_radius,
            axial_length=req.axial_length,
            hub_curvature=req.hub_curvature,
            shroud_curvature=req.shroud_curvature,
            inlet_angle_hub=req.inlet_angle_hub,
            inlet_angle_shroud=req.inlet_angle_shroud,
            outlet_angle_hub=req.outlet_angle_hub,
            outlet_angle_shroud=req.outlet_angle_shroud,
            hub_inlet_radius=req.hub_inlet_radius,
            passage_width_inlet=req.passage_width_inlet,
        )
        gen = MRIGenerator(params)
        profile = gen.generate(n_points=req.n_points)

        return ProfileResponse(
            hub_rz=profile.hub_rz.tolist(),
            shroud_rz=profile.shroud_rz.tolist(),
            n_points=profile.n_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/from_sizing", response_model=ProfileResponse)
def generate_from_sizing(req: MeridionalFromSizingRequest) -> ProfileResponse:
    """Auto-generate meridional from sizing result dimensions."""
    try:
        # Build a simple namespace so from_sizing_result can read attributes
        class _Sizing:
            pass

        s = _Sizing()
        s.impeller_d1 = req.impeller_d1  # type: ignore[attr-defined]
        s.impeller_d2 = req.impeller_d2  # type: ignore[attr-defined]
        s.impeller_b1 = req.impeller_b1  # type: ignore[attr-defined]
        s.impeller_b2 = req.impeller_b2  # type: ignore[attr-defined]
        s.shaft_diameter = req.shaft_diameter  # type: ignore[attr-defined]

        gen = MRIGenerator.from_sizing_result(s)
        profile = gen.generate(n_points=req.n_points)

        return ProfileResponse(
            hub_rz=profile.hub_rz.tolist(),
            shroud_rz=profile.shroud_rz.tolist(),
            n_points=profile.n_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/from_control_points", response_model=ControlPointsResponse)
def generate_from_control_points(req: ControlPointsRequest) -> ControlPointsResponse:
    """Interpolate smooth hub/shroud from drag-editor control points.

    Accepts ordered (r, z) control points for hub and shroud, performs
    composite cubic Bezier interpolation, and returns the smooth curves
    along with computed passage metrics.
    """
    try:
        hub = _interpolate_composite_bezier(req.hub_points, req.n_output_points)
        shroud = _interpolate_composite_bezier(req.shroud_points, req.n_output_points)
        metrics = _compute_metrics(hub, shroud)

        return ControlPointsResponse(
            hub_rz=hub.tolist(),
            shroud_rz=shroud.tolist(),
            n_points=req.n_output_points,
            metrics=metrics,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/validate_control_points", response_model=ControlPointsValidateResponse)
def validate_control_points(req: ControlPointsValidateRequest) -> ControlPointsValidateResponse:
    """Validate control-point geometry from the interactive editor."""
    try:
        hub = _interpolate_composite_bezier(req.hub_points, req.n_output_points)
        shroud = _interpolate_composite_bezier(req.shroud_points, req.n_output_points)
        valid, errors, warnings = _validate_control_points(hub, shroud)
        metrics = _compute_metrics(hub, shroud)

        return ControlPointsValidateResponse(
            valid=valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/templates/control_points", response_model=TemplateControlPointsResponse)
def get_template_control_points(req: TemplateControlPointsRequest) -> TemplateControlPointsResponse:
    """Return preset drag-editor control points for a given template.

    Generates hub/shroud curves for the template and samples 6 evenly
    spaced control points along each curve.
    """
    try:
        template = MRITemplate(req.template)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown template: {req.template}. "
                   f"Valid: {[t.value for t in MRITemplate]}",
        )

    try:
        params = template.to_params(d2=req.d2, b2=req.b2)
        gen = MRIGenerator(params)
        profile = gen.generate(n_points=60)

        # Sample 6 control points from each curve (endpoints + 4 interior)
        indices = np.linspace(0, len(profile.hub_rz) - 1, 6, dtype=int)
        hub_pts = [
            RZPoint(r=float(profile.hub_rz[i, 0]), z=float(profile.hub_rz[i, 1]))
            for i in indices
        ]
        shroud_pts = [
            RZPoint(r=float(profile.shroud_rz[i, 0]), z=float(profile.shroud_rz[i, 1]))
            for i in indices
        ]

        return TemplateControlPointsResponse(
            hub_points=hub_pts,
            shroud_points=shroud_pts,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/templates", response_model=TemplatesResponse)
def list_templates() -> TemplatesResponse:
    """List available meridional profile templates."""
    descriptions = {
        MRITemplate.RADIAL_PUMP: "Standard centrifugal pump meridional (L/D2 ~ 0.3)",
        MRITemplate.MIXED_FLOW: "Mixed-flow pump with 45-deg exit (L/D2 ~ 0.5)",
        MRITemplate.FRANCIS_TURBINE: "Francis turbine — high wrap, wide inlet (L/D2 ~ 0.7)",
        MRITemplate.AXIAL: "Axial machine — cylindrical annulus (constant radius)",
    }
    templates = [
        TemplateInfo(name=t.value, description=descriptions.get(t, ""))
        for t in MRITemplate
    ]
    return TemplatesResponse(templates=templates)


@router.post("/validate", response_model=ValidationResponse)
def validate_meridional(req: MeridionalValidateRequest) -> ValidationResponse:
    """Validate meridional geometry for intersections and curvature."""
    try:
        params = MRIParams(
            inlet_radius=req.inlet_radius,
            outlet_radius=req.outlet_radius,
            axial_length=req.axial_length,
            hub_curvature=req.hub_curvature,
            shroud_curvature=req.shroud_curvature,
            inlet_angle_hub=req.inlet_angle_hub,
            inlet_angle_shroud=req.inlet_angle_shroud,
            outlet_angle_hub=req.outlet_angle_hub,
            outlet_angle_shroud=req.outlet_angle_shroud,
            hub_inlet_radius=req.hub_inlet_radius,
            passage_width_inlet=req.passage_width_inlet,
        )
        gen = MRIGenerator(params)
        result = gen.validate(n_points=req.n_points)

        return ValidationResponse(
            valid=result.valid,
            errors=result.errors,
            warnings=result.warnings,
            min_passage_width=result.min_passage_width,
            max_curvature_hub=result.max_curvature_hub,
            max_curvature_shroud=result.max_curvature_shroud,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
