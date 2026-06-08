"""High-level geometry export for the HPE pipeline.

Provides a single entry point that generates a 3D runner solid (via CadQuery)
and exports it to STEP and/or STL, falling back gracefully when CadQuery is
not installed.

Usage
-----
    from hpe.geometry.export import export_runner_3d, CadExportResult

    result = export_runner_3d(sizing_result, output_dir=Path("/tmp/runner"))
    if result.available:
        print(result.step_path, result.stl_path)
    else:
        print("CadQuery not installed — 2D profiles only")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    import cadquery as cq
    _CQ_AVAILABLE = True
except ImportError:
    cq = None  # type: ignore[assignment]
    _CQ_AVAILABLE = False


@dataclass
class CadExportResult:
    """Result of a 3D geometry export attempt.

    Attributes
    ----------
    available : bool
        True if CadQuery was installed and the solid was generated.
    step_path : Optional[Path]
        Path to the exported STEP file, or None if not generated.
    stl_path : Optional[Path]
        Path to the exported STL file, or None if not generated.
    reason : str
        Human-readable explanation when available=False.
    """
    available: bool
    step_path: Optional[Path] = None
    stl_path: Optional[Path] = None
    reason: str = ""


def export_runner_3d(
    sizing_result,
    output_dir: Path,
    export_step: bool = True,
    export_stl: bool = True,
    stl_tolerance: float = 0.01,
    stl_angular_tolerance: float = 0.1,
) -> CadExportResult:
    """Generate and export a 3D runner solid from a SizingResult.

    This function:

    1. Converts ``sizing_result`` into ``RunnerGeometryParams``.
    2. Calls the CadQuery runner builder (``hpe.geometry.runner.builder``).
    3. Exports to STEP and/or STL in ``output_dir``.

    When CadQuery is not installed the function returns immediately with
    ``CadExportResult(available=False, reason=...)``.

    Args:
        sizing_result: A ``SizingResult`` instance from ``hpe.sizing``.
        output_dir: Directory to write output files into (created if missing).
        export_step: Whether to export a STEP file.
        export_stl: Whether to export an STL file.
        stl_tolerance: STL linear tessellation tolerance [mm].
        stl_angular_tolerance: STL angular tessellation tolerance [degrees].

    Returns:
        :class:`CadExportResult` with paths to generated files.
    """
    if not _CQ_AVAILABLE:
        reason = (
            "CadQuery is not installed.  Run with the backend-cad Docker image "
            "or install via: pip install cadquery>=2.4"
        )
        log.warning("export_runner_3d: %s", reason)
        return CadExportResult(available=False, reason=reason)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build RunnerGeometryParams from SizingResult ---
    try:
        from hpe.geometry.models import RunnerGeometryParams
        params = RunnerGeometryParams.from_sizing(sizing_result)
    except Exception as exc:
        reason = f"Failed to build RunnerGeometryParams: {exc}"
        log.error("export_runner_3d: %s", reason)
        return CadExportResult(available=False, reason=reason)

    # --- Generate 3D solid ---
    try:
        from hpe.geometry.runner.builder import build_runner_solid
        solid = build_runner_solid(params)
    except Exception as exc:
        reason = f"CadQuery solid generation failed: {exc}"
        log.error("export_runner_3d: %s", reason, exc_info=True)
        return CadExportResult(available=False, reason=reason)

    step_path: Optional[Path] = None
    stl_path: Optional[Path] = None

    # --- STEP export ---
    if export_step:
        try:
            step_path = output_dir / "runner.step"
            cq.exporters.export(solid, str(step_path), exportType="STEP")
            log.info("export_runner_3d: STEP written to %s", step_path)
        except Exception as exc:
            log.error("export_runner_3d: STEP export failed: %s", exc)
            step_path = None

    # --- STL export ---
    if export_stl:
        try:
            stl_path = output_dir / "runner.stl"
            cq.exporters.export(
                solid,
                str(stl_path),
                exportType="STL",
                tolerance=stl_tolerance,
                angularTolerance=stl_angular_tolerance,
            )
            log.info("export_runner_3d: STL written to %s", stl_path)
        except Exception as exc:
            log.error("export_runner_3d: STL export failed: %s", exc)
            stl_path = None

    return CadExportResult(
        available=True,
        step_path=step_path,
        stl_path=stl_path,
    )
