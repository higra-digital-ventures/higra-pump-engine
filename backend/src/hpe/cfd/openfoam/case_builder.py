"""OpenFOAM case builder — assembles a complete case from sizing + geometry.

Takes a SizingResult and a STEP file, and produces a ready-to-run
OpenFOAM case directory with all configuration files.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from hpe.cfd.mesh.blockmesh import generate_blockmesh_dict
from hpe.cfd.mesh.snappy import generate_snappy_dict
from hpe.cfd.openfoam.boundary_conditions import (
    BCValues,
    calc_bc_values,
    generate_mrf_properties,
)
from hpe.core.models import SizingResult


def build_case(
    sizing: SizingResult,
    step_file: Path | str,
    output_dir: Path | str,
    flow_rate: float | None = None,
    n_procs: int = 4,
) -> Path:
    """Build a complete OpenFOAM case directory.

    Pipeline:
        1. Create directory structure (0/, constant/, system/)
        2. Copy templates
        3. Generate blockMeshDict
        4. Generate snappyHexMeshDict
        5. Calculate and apply boundary conditions
        6. Generate MRFProperties
        7. Copy geometry to constant/triSurface/
        8. Generate run script

    Args:
        sizing: SizingResult from the sizing module.
        step_file: Path to the STEP file from geometry module.
        output_dir: Where to create the case.
        flow_rate: Override flow rate [m3/s]. If None, uses design point.
        n_procs: Number of processors for parallel run.

    Returns:
        Path to the case directory.
    """
    step_file = Path(step_file)
    output_dir = Path(output_dir)

    # Determine flow rate
    if flow_rate is None:
        from hpe.physics.euler import get_design_flow_rate
        flow_rate = get_design_flow_rate(sizing)

    # Extract geometry data
    mp = sizing.meridional_profile
    d_inlet = sizing.impeller_d1
    d_hub = mp.get("d1_hub", d_inlet * 0.35)

    # RPM from velocity triangles
    u2 = sizing.velocity_triangles["outlet"]["u"]
    rpm = 60.0 * u2 / (math.pi * sizing.impeller_d2)

    # 1. Create directory structure
    _create_dirs(output_dir)

    # 2. Copy templates
    _copy_templates(output_dir)

    # 3. Generate blockMeshDict
    blockmesh = generate_blockmesh_dict(sizing.impeller_d2, sizing.impeller_b2)
    _write_file(output_dir / "system" / "blockMeshDict", blockmesh)

    # 4. Generate snappyHexMeshDict
    # Convert STEP to STL name (snappyHexMesh uses STL)
    stl_name = "impeller.stl"
    snappy = generate_snappy_dict(stl_name, sizing.impeller_d2)
    _write_file(output_dir / "system" / "snappyHexMeshDict", snappy)

    # 5. Calculate BCs and apply
    bc = calc_bc_values(flow_rate, rpm, d_inlet, d_hub)
    _apply_boundary_conditions(output_dir, bc, n_procs)

    # 6. MRF
    mrf = generate_mrf_properties(bc.omega_rotor)
    _write_file(output_dir / "constant" / "MRFProperties", mrf)

    # 7. Copy geometry
    tri_dir = output_dir / "constant" / "triSurface"
    tri_dir.mkdir(parents=True, exist_ok=True)
    if step_file.exists():
        shutil.copy2(step_file, tri_dir / step_file.name)

    # 8. Generate run script
    _generate_run_script(output_dir, n_procs)

    return output_dir


def _create_dirs(case_dir: Path) -> None:
    for d in ["0", "constant", "system", "constant/triSurface"]:
        (case_dir / d).mkdir(parents=True, exist_ok=True)


def _copy_templates(case_dir: Path) -> None:
    """Copy static template files that don't need substitution."""
    template_dir = Path(__file__).parents[3] / "data" / "templates" / "openfoam" / "centrifugal_pump"

    if not template_dir.exists():
        # Fallback: try from working directory
        template_dir = Path("data/templates/openfoam/centrifugal_pump")

    if not template_dir.exists():
        return  # Templates not found, files will be generated instead

    # Copy files that don't have template variables
    static_files = [
        ("system/controlDict", "system/controlDict"),
        ("system/fvSchemes", "system/fvSchemes"),
        ("system/fvSolution", "system/fvSolution"),
        ("constant/momentumTransport", "constant/momentumTransport"),
    ]

    for src_rel, dst_rel in static_files:
        src = template_dir / src_rel
        dst = case_dir / dst_rel
        if src.exists():
            shutil.copy2(src, dst)


def _apply_boundary_conditions(case_dir: Path, bc: BCValues, n_procs: int) -> None:
    """Write BC files with computed values."""
    template_dir = Path(__file__).parents[3] / "data" / "templates" / "openfoam" / "centrifugal_pump"
    if not template_dir.exists():
        template_dir = Path("data/templates/openfoam/centrifugal_pump")

    substitutions = {
        "{{U_INLET}}": f"{bc.u_inlet:.6f}",
        "{{K_INIT}}": f"{bc.k_init:.8f}",
        "{{OMEGA_INIT}}": f"{bc.omega_turb_init:.4f}",
        "{{NU}}": f"{bc.nu:.8e}",
        "{{N_PROCS}}": str(n_procs),
    }

    template_files = [
        "0/U", "0/p", "0/k", "0/omega",
        "constant/transportProperties",
        "system/decomposeParDict",
    ]

    for rel_path in template_files:
        src = template_dir / rel_path
        if src.exists():
            content = src.read_text()
            for key, value in substitutions.items():
                content = content.replace(key, value)
            _write_file(case_dir / rel_path, content)


def _generate_run_script(case_dir: Path, n_procs: int) -> None:
    """Generate run.sh script for executing the case."""
    script = """#!/bin/bash
# HPE OpenFOAM run script
# Generated by Higra Pump Engine

set -e

echo "=== HPE: Starting OpenFOAM simulation ==="

# 1. Background mesh
echo "--- Step 1: blockMesh ---"
blockMesh

# 2. Decompose for parallel (if n_procs > 1)
"""
    if n_procs > 1:
        script += f"""
echo "--- Step 2: surfaceFeatureExtract ---"
surfaceFeatureExtract 2>/dev/null || true

echo "--- Step 3: snappyHexMesh ---"
snappyHexMesh -overwrite

echo "--- Step 4: decomposePar ---"
decomposePar -force

echo "--- Step 5: simpleFoam (parallel, {n_procs} procs) ---"
mpirun -np {n_procs} simpleFoam -parallel

echo "--- Step 6: reconstructPar ---"
reconstructPar -latestTime
"""
    else:
        script += """
echo "--- Step 2: surfaceFeatureExtract ---"
surfaceFeatureExtract 2>/dev/null || true

echo "--- Step 3: snappyHexMesh ---"
snappyHexMesh -overwrite

echo "--- Step 4: simpleFoam (serial) ---"
simpleFoam
"""

    script += """
echo "=== HPE: Simulation complete ==="
"""

    run_script = case_dir / "run.sh"
    _write_file(run_script, script)
    run_script.chmod(0o755)


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
