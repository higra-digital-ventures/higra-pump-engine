"""PDF report generation endpoint (#18) and .hpe project file endpoints (I1)."""

from __future__ import annotations

import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1", tags=["report"])


# ── Request / Response schemas ────────────────────────────────────────────────

class ReportRequest(BaseModel):
    """Full operating point + results for report generation."""
    project_name: str = Field("Projeto HPE", description="Project title")
    flow_rate: float = Field(..., gt=0, description="Q [m³/s]")
    head: float = Field(..., gt=0, description="H [m]")
    rpm: float = Field(..., gt=0, description="N [rpm]")
    # Optional: pre-computed results (if not provided, sizing is re-run)
    sizing: Optional[Dict[str, Any]] = None
    author: str = Field("Engenharia HIGRA", description="Report author")
    notes: str = Field("", description="Additional notes")


# ── PDF builder ───────────────────────────────────────────────────────────────

_ACCENT = (0, 160, 223)        # #00A0DF
_DARK   = (20, 20, 20)
_GRAY   = (80, 80, 80)
_LIGHT  = (230, 230, 230)
_WARN   = (255, 213, 79)
_WHITE  = (255, 255, 255)
_RED    = (220, 53, 69)


def _build_pdf(req: ReportRequest, s: Dict[str, Any]) -> bytes:
    """Build and return PDF bytes from sizing result dict."""
    from fpdf import FPDF  # imported lazily to avoid hard import at startup

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    W = 174  # usable width (A4 210 - 2×18)

    # ── Helper lambdas ────────────────────────────────────────────────────────
    def set_color(rgb: tuple[int, int, int], fill: bool = False) -> None:
        if fill:
            pdf.set_fill_color(*rgb)
        else:
            pdf.set_text_color(*rgb)

    def rule(color: tuple[int, int, int] = _LIGHT, thickness: float = 0.3) -> None:
        pdf.set_draw_color(*color)
        pdf.set_line_width(thickness)
        pdf.line(18, pdf.get_y(), 18 + W, pdf.get_y())
        pdf.ln(2)

    def section_header(text: str) -> None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        set_color(_ACCENT)
        pdf.cell(0, 6, text, ln=True)
        rule(_ACCENT, 0.5)

    def kv_row(label: str, value: str, unit: str = "", warn: bool = False) -> None:
        pdf.set_font("Helvetica", "", 9)
        set_color(_GRAY)
        pdf.cell(65, 5.5, label, ln=False)
        set_color(_RED if warn else _DARK)
        pdf.set_font("Helvetica", "B" if warn else "", 9)
        pdf.cell(W - 65, 5.5, f"{value}  {unit}".strip(), ln=True)

    def two_col(pairs: list[tuple[str, str, str]]) -> None:
        """Render two columns of kv pairs."""
        col_w = W // 2
        for i in range(0, len(pairs), 2):
            row_y = pdf.get_y()
            # left
            lbl, val, unit = pairs[i]
            pdf.set_font("Helvetica", "", 9)
            set_color(_GRAY)
            pdf.set_xy(18, row_y)
            pdf.cell(30, 5.5, lbl)
            set_color(_DARK)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(col_w - 30, 5.5, f"{val}  {unit}".strip())
            # right
            if i + 1 < len(pairs):
                lbl2, val2, unit2 = pairs[i + 1]
                pdf.set_xy(18 + col_w, row_y)
                pdf.set_font("Helvetica", "", 9)
                set_color(_GRAY)
                pdf.cell(30, 5.5, lbl2)
                set_color(_DARK)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(col_w - 30, 5.5, f"{val2}  {unit2}".strip())
            pdf.ln(5.5)

    # ── COVER HEADER ─────────────────────────────────────────────────────────
    # Accent bar
    set_color(_ACCENT, fill=True)
    pdf.set_draw_color(*_ACCENT)
    pdf.rect(18, 18, W, 14, style="F")

    pdf.set_xy(18, 19)
    pdf.set_font("Helvetica", "B", 14)
    set_color(_WHITE)
    pdf.cell(W // 2, 12, "HIGRA Pump Engine", ln=False, align="L")

    pdf.set_xy(18 + W // 2, 19)
    pdf.set_font("Helvetica", "", 9)
    set_color(_WHITE)
    pdf.cell(W // 2, 6, "Relatório de Dimensionamento", ln=False, align="R")
    pdf.set_xy(18 + W // 2, 24)
    pdf.cell(W // 2, 6, datetime.now().strftime("%d/%m/%Y %H:%M"), ln=True, align="R")

    pdf.ln(6)

    # Project title
    pdf.set_font("Helvetica", "B", 13)
    set_color(_DARK)
    pdf.cell(0, 8, req.project_name, ln=True)
    pdf.set_font("Helvetica", "", 9)
    set_color(_GRAY)
    pdf.cell(0, 5, f"Elaborado por: {req.author}", ln=True)
    pdf.ln(2)
    rule()

    # ── OPERATING POINT ───────────────────────────────────────────────────────
    section_header("1. Ponto de Operação")
    q_m3h = req.flow_rate * 3600
    two_col([
        ("Vazão (Q)", f"{q_m3h:.1f}", "m³/h"),
        ("Altura (H)", f"{req.head:.1f}", "m"),
        ("Velocidade (N)", f"{req.rpm:.0f}", "rpm"),
        ("Fluido", "Água", ""),
    ])

    # ── SIZING RESULTS ─────────────────────────────────────────────────────────
    section_header("2. Resultados do Dimensionamento")

    nq = s.get("specific_speed_nq", 0)
    imp_type = s.get("impeller_type", "—")
    eta = s.get("estimated_efficiency", 0) * 100
    power_kw = s.get("estimated_power", 0) / 1000
    npsh = s.get("estimated_npsh_r", 0)
    d2_mm = s.get("impeller_d2", 0) * 1000
    d1_mm = s.get("impeller_d1", 0) * 1000
    b2_mm = s.get("impeller_b2", 0) * 1000
    z = s.get("blade_count", 0)
    beta1 = s.get("beta1", 0)
    beta2 = s.get("beta2", 0)
    sigma = s.get("sigma", 0)

    two_col([
        ("Nq (vel. específica)", f"{nq:.1f}", ""),
        ("Tipo de rotor", imp_type, ""),
        ("Efic. estimada (η)", f"{eta:.1f}", "%"),
        ("Potência no eixo", f"{power_kw:.2f}", "kW"),
        ("NPSHr", f"{npsh:.2f}", "m"),
        ("Coef. cavitação (σ)", f"{sigma:.4f}", ""),
    ])

    section_header("3. Geometria do Rotor")
    two_col([
        ("D2 (diâm. externo)", f"{d2_mm:.1f}", "mm"),
        ("D1 (diâm. interno)", f"{d1_mm:.1f}", "mm"),
        ("b2 (largura saída)", f"{b2_mm:.1f}", "mm"),
        ("Número de pás (Z)", f"{z}", ""),
        ("β1 (ângulo entrada)", f"{beta1:.1f}", "°"),
        ("β2 (ângulo saída)", f"{beta2:.1f}", "°"),
    ])

    # ── UNCERTAINTY BOUNDS ────────────────────────────────────────────────────
    unc = s.get("uncertainty", {})
    if unc:
        section_header("4. Incerteza das Correlações")
        two_col([
            ("ΔD2", f"±{unc.get('d2_pct', 0):.0f}", "%"),
            ("Δη", f"±{unc.get('eta_pct', 0):.0f}", "%"),
            ("ΔNPSHr", f"±{unc.get('npsh_pct', 0):.0f}", "%"),
            ("Δb2", f"±{unc.get('b2_pct', 0):.0f}", "%"),
        ])
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 8)
        set_color(_GRAY)
        pdf.multi_cell(0, 4.5,
            "Incertezas baseadas em dispersão estatística das correlações de Gülich (2014), "
            "Stepanoff (1957) e Pfleiderer (1961). Para dimensionamento final, recomenda-se "
            "validação por CFD e ensaio em bancada.",
            ln=True,
        )

    # ── TRIANGULOS DE VELOCIDADE ──────────────────────────────────────────────
    vt = s.get("velocity_triangles", {})
    if vt:
        section_header(f"{'5' if unc else '4'}. Triângulos de Velocidade")
        inlet = vt.get("inlet", {})
        outlet = vt.get("outlet", {})
        sec_n = "6" if unc else "5"

        rows_in = [
            ("Velocidade meridional cm1", f"{inlet.get('cm1', 0):.2f}", "m/s"),
            ("Velocidade periférica u1",  f"{inlet.get('u1', 0):.2f}", "m/s"),
            ("Velocidade relativa w1",    f"{inlet.get('w1', 0):.2f}", "m/s"),
        ]
        rows_out = [
            ("Velocidade meridional cm2", f"{outlet.get('cm2', 0):.2f}", "m/s"),
            ("Velocidade periférica u2",  f"{outlet.get('u2', 0):.2f}", "m/s"),
            ("Velocidade relativa w2",    f"{outlet.get('w2', 0):.2f}", "m/s"),
        ]

        pdf.set_font("Helvetica", "B", 9)
        set_color(_GRAY)
        pdf.cell(0, 5, "Entrada do rotor", ln=True)
        for lbl, val, unit in rows_in:
            kv_row(lbl, val, unit)

        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 9)
        set_color(_GRAY)
        pdf.cell(0, 5, "Saída do rotor", ln=True)
        for lbl, val, unit in rows_out:
            kv_row(lbl, val, unit)
    else:
        sec_n = "5" if unc else "4"

    # ── WARNINGS ─────────────────────────────────────────────────────────────
    warnings: List[str] = s.get("warnings", [])
    if warnings:
        section_header(f"{sec_n}. Avisos")
        for w in warnings:
            pdf.set_font("Helvetica", "", 9)
            set_color(_RED)
            pdf.cell(4, 5, "•", ln=False)
            set_color(_DARK)
            pdf.multi_cell(0, 5, w, ln=True)

    # ── NOTES ─────────────────────────────────────────────────────────────────
    if req.notes.strip():
        section_header("Observações")
        pdf.set_font("Helvetica", "", 9)
        set_color(_DARK)
        pdf.multi_cell(0, 5, req.notes.strip(), ln=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    # Draw footer on every page
    total_pages = pdf.pages
    for page_num in range(1, total_pages + 1):
        pdf.page = page_num
        pdf.set_y(-14)
        rule(_LIGHT, 0.2)
        pdf.set_font("Helvetica", "I", 7)
        set_color(_GRAY)
        pdf.cell(W // 2, 4,
                 "HIGRA Industrial Ltda. — Confidencial — Uso Restrito",
                 ln=False, align="L")
        pdf.cell(W // 2, 4,
                 f"Página {page_num} / {total_pages}",
                 ln=True, align="R")

    return bytes(pdf.output())


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/report/pdf")
def generate_pdf_report(req: ReportRequest) -> Response:
    """Generate a PDF engineering report for the sizing result (#18).

    If ``req.sizing`` is not provided, re-runs sizing internally.
    Returns application/pdf binary.
    """
    from hpe.core.models import OperatingPoint
    from hpe.sizing import run_sizing

    sizing_dict = req.sizing
    if sizing_dict is None:
        op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
        result = run_sizing(op)
        unc = result.uncertainty.as_dict() if result.uncertainty else {}
        sizing_dict = {
            "specific_speed_nq": result.specific_speed_nq,
            "impeller_type": result.meridional_profile.get("impeller_type", "—"),
            "estimated_efficiency": result.estimated_efficiency,
            "estimated_power": result.estimated_power,
            "estimated_npsh_r": result.estimated_npsh_r,
            "sigma": result.sigma,
            "impeller_d2": result.impeller_d2,
            "impeller_d1": result.impeller_d1,
            "impeller_b2": result.impeller_b2,
            "blade_count": result.blade_count,
            "beta1": result.beta1,
            "beta2": result.beta2,
            "velocity_triangles": result.velocity_triangles,
            "warnings": result.warnings,
            "uncertainty": unc,
        }

    pdf_bytes = _build_pdf(req, sizing_dict)
    filename = req.project_name.replace(" ", "_") + ".pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Project file schemas ──────────────────────────────────────────────────────

class ProjectSaveRequest(BaseModel):
    """Request to save an HPE project file."""
    name: str = Field("Projeto HPE", description="Project name")
    author: str = Field("Engenharia HIGRA", description="Author name")
    flow_rate: float = Field(..., gt=0, description="Q [m³/s]")
    head: float = Field(..., gt=0, description="H [m]")
    rpm: float = Field(..., gt=0, description="N [rpm]")
    sizing: Optional[Dict[str, Any]] = None
    optimization_history: Optional[List[Any]] = Field(default_factory=list)
    notes: str = Field("", description="Additional notes")


class ProjectSaveResponse(BaseModel):
    """Response containing base64-encoded .hpe file content."""
    filename: str
    content_base64: str
    version: str
    saved_at: str


class ProjectLoadRequest(BaseModel):
    """Request to load an HPE project from base64-encoded .hpe content."""
    content_base64: str = Field(..., description="Base64-encoded .hpe file bytes")


# ── Project file endpoints ────────────────────────────────────────────────────

@router.post("/project/save", response_model=ProjectSaveResponse)
def save_project_endpoint(req: ProjectSaveRequest) -> ProjectSaveResponse:
    """Save sizing result as .hpe file and return base64-encoded content (I1).

    If sizing is not provided, re-runs sizing internally.
    Returns base64-encoded gzip-compressed JSON project file.
    """
    from hpe.core.project_file import save_project, HPE_VERSION
    from hpe.core.models import OperatingPoint

    sizing_dict = req.sizing
    if sizing_dict is None:
        from hpe.sizing import run_sizing
        op = OperatingPoint(flow_rate=req.flow_rate, head=req.head, rpm=req.rpm)
        result = run_sizing(op)
        unc = result.uncertainty.as_dict() if result.uncertainty else {}
        sizing_dict = {
            "specific_speed_nq": result.specific_speed_nq,
            "impeller_type": result.meridional_profile.get("impeller_type", "—"),
            "estimated_efficiency": result.estimated_efficiency,
            "estimated_power": result.estimated_power,
            "estimated_npsh_r": result.estimated_npsh_r,
            "sigma": result.sigma,
            "impeller_d2": result.impeller_d2,
            "impeller_d1": result.impeller_d1,
            "impeller_b2": result.impeller_b2,
            "blade_count": result.blade_count,
            "beta1": result.beta1,
            "beta2": result.beta2,
            "velocity_triangles": result.velocity_triangles,
            "warnings": result.warnings,
            "uncertainty": unc,
        }

    project_data = {
        "name": req.name,
        "author": req.author,
        "operating_point": {
            "flow_rate": req.flow_rate,
            "head": req.head,
            "rpm": req.rpm,
        },
        "sizing_result": sizing_dict,
        "optimization_history": req.optimization_history or [],
        "notes": req.notes,
    }

    with tempfile.NamedTemporaryFile(suffix=".hpe", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_project(tmp_path, project_data)
        raw_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    content_b64 = base64.b64encode(raw_bytes).decode("ascii")
    safe_name = req.name.replace(" ", "_")
    now_iso = datetime.now().isoformat()

    return ProjectSaveResponse(
        filename=f"{safe_name}.hpe",
        content_base64=content_b64,
        version=HPE_VERSION,
        saved_at=now_iso,
    )


@router.post("/project/load")
def load_project_endpoint(req: ProjectLoadRequest) -> Dict[str, Any]:
    """Load project from base64-encoded .hpe file and return project data (I1).

    Accepts a base64-encoded .hpe file, validates it, and returns the full
    project envelope including metadata and stored design data.
    """
    from hpe.core.project_file import load_project

    try:
        raw_bytes = base64.b64decode(req.content_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".hpe", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        envelope = load_project(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read project file: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return envelope
