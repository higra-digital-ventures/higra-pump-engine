"""CFD pipeline orchestrator.

Coordinates the full flow: SizingResult → Geometry export → OpenFOAM case → Run → Post.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hpe.core.models import PerformanceMetrics, SizingResult


@dataclass
class PipelineResult:
    """Result of a CFD pipeline run."""

    case_dir: Path
    step_file: Path | None
    openfoam_available: bool
    ran_simulation: bool
    performance: PerformanceMetrics | None
    errors: list[str]


def run_cfd_pipeline(
    sizing: SizingResult,
    output_dir: Path | str,
    flow_rate: float | None = None,
    run_solver: bool = True,
    export_geometry: bool = True,
    n_procs: int = 4,
) -> PipelineResult:
    """Execute the full CFD pipeline.

    Steps:
        1. Export geometry to STEP (if export_geometry=True)
        2. Build OpenFOAM case
        3. Run solver (if run_solver=True and OpenFOAM available)
        4. Post-process results

    Args:
        sizing: SizingResult from sizing module.
        output_dir: Base output directory.
        flow_rate: Override flow rate. None = design point.
        run_solver: Whether to attempt running OpenFOAM.
        export_geometry: Whether to generate STEP file.
        n_procs: Number of processors for parallel run.

    Returns:
        PipelineResult with case path and optional performance data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    # 1. Export geometry
    step_file: Optional[Path] = None
    if export_geometry:
        try:
            from hpe.core.enums import GeometryFormat
            from hpe.geometry.runner import generate_runner_from_sizing
            from hpe.geometry.runner.export import export_runner

            runner = generate_runner_from_sizing(sizing)
            step_file = export_runner(
                runner, output_dir / "impeller.step", GeometryFormat.STEP,
            )
            # Also export STL for snappyHexMesh
            export_runner(
                runner, output_dir / "impeller.stl", GeometryFormat.STL,
            )
        except Exception as e:
            errors.append(f"Geometry export failed: {e}")

    # 2. Build OpenFOAM case
    case_dir = output_dir / "case"
    try:
        from hpe.cfd.openfoam.case_builder import build_case

        stl_source = output_dir / "impeller.stl"
        build_case(
            sizing, stl_source, case_dir,
            flow_rate=flow_rate, n_procs=n_procs,
        )
    except Exception as e:
        errors.append(f"Case build failed: {e}")
        return PipelineResult(
            case_dir=case_dir, step_file=step_file,
            openfoam_available=False, ran_simulation=False,
            performance=None, errors=errors,
        )

    # 3. Check OpenFOAM and optionally run
    from hpe.cfd.openfoam.runner import check_openfoam_available

    of_available = check_openfoam_available()
    ran = False
    performance: Optional[PerformanceMetrics] = None

    if run_solver and of_available:
        try:
            from hpe.cfd.openfoam.runner import run_case

            results = run_case(case_dir, parallel=(n_procs > 1), n_procs=n_procs)
            ran = all(r.success for r in results)

            if not ran:
                for r in results:
                    if not r.success:
                        errors.append(f"{r.command}: {r.stderr[:200]}")
        except Exception as e:
            errors.append(f"Solver execution failed: {e}")

    elif run_solver and not of_available:
        errors.append(
            "OpenFOAM not installed. Case generated but not executed. "
            f"Run manually: cd {case_dir} && ./run.sh"
        )

    # 4. Post-process (if solver ran)
    if ran:
        try:
            performance = _post_process(case_dir, sizing)
        except Exception as e:
            errors.append(f"Post-processing failed: {e}")

    return PipelineResult(
        case_dir=case_dir,
        step_file=step_file,
        openfoam_available=of_available,
        ran_simulation=ran,
        performance=performance,
        errors=errors,
    )


def _post_process(case_dir: Path, sizing: SizingResult) -> Optional[PerformanceMetrics]:
    """Extract performance from completed simulation."""
    import math

    from hpe.postprocess.openfoam_parser import parse_forces

    forces_dir = case_dir / "postProcessing" / "forces1"
    forces = parse_forces(forces_dir)

    if not forces.moment_z:
        return None

    from hpe.postprocess.metrics import CFDMetrics, calc_performance_from_cfd

    u2 = sizing.velocity_triangles["outlet"]["u"]
    rpm = 60.0 * u2 / (math.pi * sizing.impeller_d2)

    from hpe.physics.euler import get_design_flow_rate

    cfd = CFDMetrics(
        torque_z=forces.moment_z[-1] if forces.moment_z else 0,
        force_radial=math.sqrt(
            (forces.force_x[-1] if forces.force_x else 0) ** 2
            + (forces.force_y[-1] if forces.force_y else 0) ** 2
        ),
        pressure_inlet=0.0,
        pressure_outlet=0.0,
        flow_rate=get_design_flow_rate(sizing),
    )

    return calc_performance_from_cfd(cfd, rpm)
