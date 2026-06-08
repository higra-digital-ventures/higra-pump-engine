"""Gerador de relatório técnico PDF — Fase 20.4.

Gera relatórios de engenharia em PDF com:
  - Capa: nome do projeto, datas, versão HPE
  - Ponto de operação (Q, H, n, NPSHa)
  - Resultados sizing 1D (D₂, b₂, β, blade count, η, P)
  - Curvas H-Q, η-Q, NPSHr-Q (SVG embebido via reportlab se disponível)
  - Análise de cavitação (gauge + recomendações)
  - Blade loading midspan + spanwise
  - Loss audit (pie chart por zona)
  - Validação contra benchmarks
  - Apêndice: warnings + referências

Estratégia:
  - Se reportlab disponível → PDF nativo
  - Senão → HTML stand-alone (para conversão posterior via wkhtmltopdf)
  - Fallback garantido: markdown + JSON

Usage
-----
    from hpe.reports.generator import generate_report, ReportContext

    ctx = ReportContext(
        project_name="Bomba Higra 50-30",
        sizing=sizing, curves=curves, cavitation=cav,
    )
    path = generate_report(ctx, output_path="report.pdf")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm  # noqa: F401
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (  # noqa: F401
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib import colors
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


@dataclass
class ReportContext:
    """Contexto completo para geração do relatório."""
    project_name: str = "HPE Pump Design Report"
    author: str = "HPE v1.0"
    date: str = ""

    sizing: Optional[dict] = None
    curves: Optional[dict] = None
    cavitation: Optional[dict] = None
    blade_loading: Optional[dict] = None
    spanwise_loading: Optional[dict] = None
    loss_audit: Optional[dict] = None
    pulsations: Optional[dict] = None
    radial_forces: Optional[dict] = None
    benchmark_results: Optional[list] = None
    warnings: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M")


def generate_report(
    ctx: ReportContext,
    output_path: "str | Path",
    format: str = "auto",
) -> Path:
    """Gerar relatório no melhor formato disponível.

    Parameters
    ----------
    ctx : ReportContext
        Contexto com todos os dados.
    output_path : Path
        Caminho de saída (a extensão é ajustada automaticamente).
    format : str
        'auto' | 'pdf' | 'html' | 'markdown'.
    """
    output_path = Path(output_path)

    if format == "auto":
        if _REPORTLAB:
            format = "pdf"
        else:
            format = "html"

    if format == "pdf" and _REPORTLAB:
        p = output_path.with_suffix(".pdf")
        _generate_pdf(ctx, p)
        return p

    if format == "html":
        p = output_path.with_suffix(".html")
        _generate_html(ctx, p)
        return p

    p = output_path.with_suffix(".md")
    _generate_markdown(ctx, p)
    return p


# ---------------------------------------------------------------------------
# PDF (reportlab)
# ---------------------------------------------------------------------------

def _generate_pdf(ctx: ReportContext, path: Path) -> None:
    """Gerar PDF usando reportlab."""
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story: list = []

    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=20, textColor=HexColor("#1e40af"),
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=HexColor("#1e40af"))

    # ── Capa ───────────────────────────────────────────────────────────────
    story.append(Paragraph(ctx.project_name, title_style))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"<i>Relatório Técnico — {ctx.date}</i>", styles["Normal"]))
    story.append(Paragraph(f"Gerado por: {ctx.author}", styles["Normal"]))
    story.append(Spacer(1, 1 * cm))

    # ── Sizing 1D ─────────────────────────────────────────────────────────
    if ctx.sizing:
        story.append(Paragraph("1. Dimensionamento 1D (Meanline Gülich)", h2))
        story.append(_sizing_table(ctx.sizing))
        story.append(Spacer(1, 0.5 * cm))

    # ── Ponto de operação e curvas ────────────────────────────────────────
    if ctx.curves:
        story.append(Paragraph("2. Curvas Características", h2))
        story.append(Paragraph(
            f"Curva H-Q, η-Q, NPSHr-Q geradas a partir de "
            f"{len(ctx.curves.get('Q_pts', []))} pontos. BEP: "
            f"Q={ctx.curves.get('bep', {}).get('Q', '?')} m³/s, "
            f"H={ctx.curves.get('bep', {}).get('H', '?')} m, "
            f"η={ctx.curves.get('bep', {}).get('eta', '?'):.3f}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.5 * cm))

    # ── Cavitação ─────────────────────────────────────────────────────────
    if ctx.cavitation:
        story.append(Paragraph("3. Análise de Cavitação", h2))
        cav = ctx.cavitation
        story.append(Paragraph(
            f"NPSHr = {cav.get('npsh_r', 0):.2f} m, "
            f"NPSHa = {cav.get('npsh_a', 0):.2f} m, "
            f"margem = {cav.get('margin', 0):.2f} m. "
            f"Nível de risco: <b>{cav.get('risk_level', 'unknown')}</b>",
            styles["Normal"],
        ))
        for rec in cav.get("recommendations", [])[:5]:
            story.append(Paragraph(f"• {rec}", styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

    # ── Loss audit ────────────────────────────────────────────────────────
    if ctx.loss_audit:
        story.append(Paragraph("4. Auditoria de Perdas (por zona)", h2))
        story.append(_loss_audit_table(ctx.loss_audit))
        story.append(Spacer(1, 0.5 * cm))

    # ── Forças radiais ────────────────────────────────────────────────────
    if ctx.radial_forces:
        rf = ctx.radial_forces
        story.append(Paragraph("5. Forças Radiais (Stepanoff Kr)", h2))
        story.append(Paragraph(
            f"Kr médio: {rf.get('mean_kr', 0):.4f}, Kr máximo: {rf.get('max_kr', 0):.4f}. "
            f"Risco: <b>{rf.get('risk_level', '?')}</b>. "
            f"Força radial RMS: {rf.get('rms_force_N', 0):.1f} N. "
            f"BPF: {rf.get('bpf_hz', 0):.1f} Hz — amp {rf.get('bpf_amplitude_N', 0):.1f} N",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.5 * cm))

    # ── Validação ─────────────────────────────────────────────────────────
    if ctx.benchmark_results:
        story.append(Paragraph("6. Validação contra Benchmarks", h2))
        rows = [["Benchmark", "N pts", "MAPE H (%)", "MAPE η (%)", "Status"]]
        for r in ctx.benchmark_results:
            rows.append([
                r.get("benchmark", "?"),
                str(r.get("n_points", 0)),
                f"{r.get('mape_head_pct', 0):.2f}",
                f"{r.get('mape_efficiency_pct', 0):.2f}",
                "PASS" if r.get("passed") else "FAIL",
            ])
        t = Table(rows)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

    # ── Warnings ──────────────────────────────────────────────────────────
    if ctx.warnings:
        story.append(Paragraph("7. Avisos", h2))
        for w in ctx.warnings:
            story.append(Paragraph(f"⚠ {w}", styles["Normal"]))

    doc.build(story)
    log.info("PDF report generated: %s", path)


def _sizing_table(sizing: dict) -> Any:
    """Criar tabela reportlab com dados do sizing."""
    rows = [
        ["Parâmetro", "Valor"],
        ["Vazão BEP [m³/h]", f"{sizing.get('Q', 0) * 3600:.1f}"],
        ["Altura BEP [m]", f"{sizing.get('H', 0):.1f}"],
        ["Rotação [rpm]", f"{sizing.get('n', 0):.0f}"],
        ["ns (specific speed)", f"{sizing.get('specific_speed_nq', 0):.1f}"],
        ["D₂ [mm]", f"{sizing.get('impeller_d2', 0) * 1000:.0f}"],
        ["D₁ [mm]", f"{sizing.get('impeller_d1', 0) * 1000:.0f}"],
        ["b₂ [mm]", f"{sizing.get('impeller_b2', 0) * 1000:.1f}"],
        ["β₁ [°]", f"{sizing.get('beta1', 0):.1f}"],
        ["β₂ [°]", f"{sizing.get('beta2', 0):.1f}"],
        ["N pás", f"{sizing.get('blade_count', 0)}"],
        ["η estimada", f"{sizing.get('estimated_efficiency', 0) * 100:.1f}%"],
        ["Potência estimada [kW]", f"{sizing.get('estimated_power', 0) / 1000:.1f}"],
    ]
    t = Table(rows, colWidths=[8 * cm, 5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
    ]))
    return t


def _loss_audit_table(audit: dict) -> Any:
    rows = [["Zona", "Potência [W]", "Head [m]", "Fração"]]
    for name, zone in audit.get("zones", {}).items():
        rows.append([
            name.title(),
            f"{zone.get('loss_power_W', 0):.1f}",
            f"{zone.get('loss_head_m', 0):.3f}",
            f"{zone.get('fraction_of_total', 0) * 100:.1f}%",
        ])
    t = Table(rows, colWidths=[4 * cm, 4 * cm, 3 * cm, 2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    return t


# ---------------------------------------------------------------------------
# HTML fallback
# ---------------------------------------------------------------------------

def _generate_html(ctx: ReportContext, path: Path) -> None:
    """HTML stand-alone, convertível via wkhtmltopdf."""
    sections = []
    sections.append(f"<h1>{ctx.project_name}</h1>")
    sections.append(f"<p><em>{ctx.date} — {ctx.author}</em></p>")

    if ctx.sizing:
        sections.append("<h2>1. Dimensionamento 1D</h2>")
        sections.append("<table>")
        for k, v in ctx.sizing.items():
            sections.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        sections.append("</table>")

    if ctx.cavitation:
        c = ctx.cavitation
        sections.append("<h2>2. Cavitação</h2>")
        sections.append(
            f"<p>NPSHr = {c.get('npsh_r', 0):.2f} m, "
            f"NPSHa = {c.get('npsh_a', 0):.2f} m, "
            f"risco = <b>{c.get('risk_level')}</b></p>"
        )

    if ctx.loss_audit:
        sections.append("<h2>3. Loss Audit</h2>")
        sections.append("<table><thead><tr><th>Zona</th><th>W</th><th>m</th><th>%</th></tr></thead><tbody>")
        for name, z in ctx.loss_audit.get("zones", {}).items():
            sections.append(
                f"<tr><td>{name}</td><td>{z.get('loss_power_W', 0):.1f}</td>"
                f"<td>{z.get('loss_head_m', 0):.3f}</td>"
                f"<td>{z.get('fraction_of_total', 0) * 100:.1f}</td></tr>"
            )
        sections.append("</tbody></table>")

    if ctx.warnings:
        sections.append("<h2>Avisos</h2><ul>")
        for w in ctx.warnings:
            sections.append(f"<li>{w}</li>")
        sections.append("</ul>")

    html = """<!doctype html><html><head><meta charset="utf-8"><title>%s</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; color: #222; }
h1 { color: #1e40af; border-bottom: 2px solid #1e40af; }
h2 { color: #1e40af; margin-top: 1.5em; }
table { border-collapse: collapse; width: 100%%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #1e40af; color: white; }
</style></head><body>
%s
</body></html>
""" % (ctx.project_name, "\n".join(sections))
    path.write_text(html, encoding="utf-8")
    log.info("HTML report generated: %s", path)


# ---------------------------------------------------------------------------
# Markdown fallback
# ---------------------------------------------------------------------------

def _generate_markdown(ctx: ReportContext, path: Path) -> None:
    out = [
        f"# {ctx.project_name}",
        f"*{ctx.date} — {ctx.author}*",
        "",
    ]
    if ctx.sizing:
        out.append("## 1. Dimensionamento 1D")
        for k, v in ctx.sizing.items():
            out.append(f"- **{k}**: {v}")
        out.append("")
    if ctx.cavitation:
        c = ctx.cavitation
        out += [
            "## 2. Cavitação",
            f"- NPSHr: {c.get('npsh_r', 0):.2f} m",
            f"- NPSHa: {c.get('npsh_a', 0):.2f} m",
            f"- Risco: **{c.get('risk_level')}**",
            "",
        ]
    if ctx.loss_audit:
        out.append("## 3. Loss Audit")
        out.append("| Zona | Potência [W] | Head [m] | Fração |")
        out.append("|---|---:|---:|---:|")
        for name, z in ctx.loss_audit.get("zones", {}).items():
            out.append(
                f"| {name} | {z.get('loss_power_W', 0):.1f} | "
                f"{z.get('loss_head_m', 0):.3f} | "
                f"{z.get('fraction_of_total', 0) * 100:.1f}% |"
            )
    path.write_text("\n".join(out), encoding="utf-8")
    log.info("Markdown report: %s", path)
