"""HPE Geometry — pipeline paramétrico público (Fase 1).

Entry point para geração de geometria a partir de resultados de sizing.
Funciona em dois modos:

  Modo 2D (sempre disponível)
      Retorna perfis meridional e de palheta como listas de pontos (r, z)
      e (r, theta). Não requer CadQuery.

  Modo 3D (requer CadQuery instalado)
      Gera sólido 3D do rotor e exporta para STEP/STL.
      Activado automaticamente quando `cadquery` está disponível.

Usage
-----
    from hpe.geometry.parametric import run_geometry, GeometryResult
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=0.05, head=30, rpm=1750)
    sizing = run_sizing(op)
    geo = run_geometry(sizing)

    print(geo.summary())           # texto com parâmetros principais
    print(geo.cad_available)       # True se CadQuery está instalado
    if geo.step_path:
        print("STEP:", geo.step_path)

API endpoint exposição
----------------------
    POST /geometry/run
    Body: SizingInput
    Returns: GeometryOutput (parâmetros + perfis 2D + step_path opcional)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hpe.core.models import SizingResult
from hpe.geometry.models import (
    BladeProfile,
    MeridionalChannel,
    RunnerGeometryParams,
)
from hpe.geometry.runner.blade import generate_blade_profile
from hpe.geometry.runner.meridional import generate_meridional_channel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CadQuery availability probe (import once, no repeated try/except)
# ---------------------------------------------------------------------------
try:
    import cadquery as _cq  # noqa: F401
    _CADQUERY_AVAILABLE = True
except ImportError:
    _CADQUERY_AVAILABLE = False
    log.debug("CadQuery not installed — 3D export disabled (2D profiles available)")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeometryResult:
    """Complete geometry result from the parametric pipeline.

    Attributes
    ----------
    params : RunnerGeometryParams
        All geometric parameters derived from sizing.
    meridional : MeridionalChannel
        Hub and shroud point clouds in (r, z) [m].
    blade : BladeProfile
        Camber line + pressure/suction side in (r, theta) [m, rad].
    cad_available : bool
        True if CadQuery is installed and 3D generation was attempted.
    step_path : str | None
        Path to exported STEP file, or None if not generated.
    stl_path : str | None
        Path to exported STL file, or None.
    generation_time_ms : float
        Wall-clock time for the full pipeline [ms].
    warnings : list[str]
        Non-fatal warnings collected during generation.
    """

    params: RunnerGeometryParams
    meridional: MeridionalChannel
    blade: BladeProfile
    cad_available: bool = False
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    generation_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-page summary of the geometry."""
        p = self.params
        lines = [
            "=== HPE Runner Geometry ===",
            f"  D2  = {p.d2 * 1000:.1f} mm   (outlet diameter)",
            f"  D1  = {p.d1 * 1000:.1f} mm   (inlet eye diameter)",
            f"  D1h = {p.d1_hub * 1000:.1f} mm   (hub diameter @ inlet)",
            f"  b2  = {p.b2 * 1000:.1f} mm   (outlet width)",
            f"  b1  = {p.b1 * 1000:.1f} mm   (inlet width)",
            f"  beta1 = {p.beta1:.1f} deg  (inlet blade angle)",
            f"  beta2 = {p.beta2:.1f} deg  (outlet blade angle)",
            f"  Z   = {p.blade_count}  blades",
            f"  t   = {p.blade_thickness * 1000:.1f} mm  (blade thickness)",
            f"  Wrap angle = {self.blade_wrap_deg:.1f} deg",
            f"  CAD export : {'STEP + STL' if self.step_path else 'not available'}",
            f"  Generated in {self.generation_time_ms:.1f} ms",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        return "\n".join(lines)

    @property
    def blade_wrap_deg(self) -> float:
        """Blade wrap angle from camber line start to end [deg]."""
        pts = self.blade.camber_points
        if len(pts) < 2:
            return 0.0
        return math.degrees(abs(pts[-1][1] - pts[0][1]))

    def to_dict(self) -> dict:
        """JSON-serialisable representation (for API responses)."""
        p = self.params
        return {
            "params": {
                "D2_mm": round(p.d2 * 1000, 1),
                "D1_mm": round(p.d1 * 1000, 1),
                "D1_hub_mm": round(p.d1_hub * 1000, 1),
                "b2_mm": round(p.b2 * 1000, 1),
                "b1_mm": round(p.b1 * 1000, 1),
                "beta1_deg": round(p.beta1, 2),
                "beta2_deg": round(p.beta2, 2),
                "blade_count": p.blade_count,
                "blade_thickness_mm": round(p.blade_thickness * 1000, 2),
                "wrap_angle_deg": round(self.blade_wrap_deg, 1),
            },
            "meridional": {
                "hub_r_mm": [round(pt[0] * 1000, 2) for pt in self.meridional.hub_points],
                "hub_z_mm": [round(pt[1] * 1000, 2) for pt in self.meridional.hub_points],
                "shroud_r_mm": [round(pt[0] * 1000, 2) for pt in self.meridional.shroud_points],
                "shroud_z_mm": [round(pt[1] * 1000, 2) for pt in self.meridional.shroud_points],
            },
            "blade": {
                "camber_r_mm": [round(pt[0] * 1000, 2) for pt in self.blade.camber_points],
                "camber_theta_deg": [round(math.degrees(pt[1]), 3) for pt in self.blade.camber_points],
                "ps_r_mm": [round(pt[0] * 1000, 2) for pt in self.blade.pressure_side],
                "ps_theta_deg": [round(math.degrees(pt[1]), 3) for pt in self.blade.pressure_side],
                "ss_r_mm": [round(pt[0] * 1000, 2) for pt in self.blade.suction_side],
                "ss_theta_deg": [round(math.degrees(pt[1]), 3) for pt in self.blade.suction_side],
            },
            "cad_available": self.cad_available,
            "step_path": self.step_path,
            "stl_path": self.stl_path,
            "generation_time_ms": round(self.generation_time_ms, 1),
            "warnings": self.warnings,
        }

    def export_json(self, path: str | Path) -> Path:
        """Save geometry as JSON (profiles + parameters)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------

def run_geometry(
    sizing: SizingResult,
    export_dir: Optional[str | Path] = None,
    export_step: bool = False,
    export_stl: bool = False,
    n_meridional_pts: int = 30,
    n_blade_pts: int = 50,
) -> GeometryResult:
    """Generate parametric runner geometry from sizing result.

    Parameters
    ----------
    sizing : SizingResult
        Output of `hpe.sizing.meanline.run_sizing()`.
    export_dir : str | Path, optional
        Directory for CAD file output.  Required if export_step or export_stl.
    export_step : bool
        Export STEP file if CadQuery is available.
    export_stl : bool
        Export STL file if CadQuery is available.
    n_meridional_pts : int
        Number of points along meridional hub/shroud curves.
    n_blade_pts : int
        Number of points along blade camber line.

    Returns
    -------
    GeometryResult
        Contains params, 2D profiles, and optionally CAD paths.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    # 1. Build geometric parameters from sizing
    params = RunnerGeometryParams.from_sizing_result(sizing)

    # 2. Generate 2D profiles (always available)
    meridional = generate_meridional_channel(params, n_points=n_meridional_pts)
    blade = generate_blade_profile(params, n_points=n_blade_pts)

    # 3. Optional 3D CAD generation
    step_path: Optional[str] = None
    stl_path: Optional[str] = None

    if (export_step or export_stl) and export_dir is not None:
        if not _CADQUERY_AVAILABLE:
            warnings.append(
                "CadQuery not installed — 3D export skipped. "
                "Install with: pip install cadquery"
            )
        else:
            try:
                from hpe.geometry.runner.impeller import generate_runner
                from hpe.geometry.runner.export import export_runner
                from hpe.core.enums import GeometryFormat

                runner_solid = generate_runner(params)
                export_dir = Path(export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)

                if export_step:
                    p = export_runner(
                        runner_solid,
                        export_dir / "runner.step",
                        fmt=GeometryFormat.STEP,
                    )
                    step_path = str(p)

                if export_stl:
                    p = export_runner(
                        runner_solid,
                        export_dir / "runner.stl",
                        fmt=GeometryFormat.STL,
                    )
                    stl_path = str(p)

                log.info("Geometry: 3D export complete — %s", export_dir)

            except Exception as exc:
                warnings.append(f"3D generation failed: {exc}")
                log.warning("Geometry: 3D export failed — %s", exc)

    elif (export_step or export_stl) and export_dir is None:
        warnings.append("export_dir required for CAD file output — skipped.")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return GeometryResult(
        params=params,
        meridional=meridional,
        blade=blade,
        cad_available=_CADQUERY_AVAILABLE,
        step_path=step_path,
        stl_path=stl_path,
        generation_time_ms=round(elapsed_ms, 2),
        warnings=warnings,
    )
