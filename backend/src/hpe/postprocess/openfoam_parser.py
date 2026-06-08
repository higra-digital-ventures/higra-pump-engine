"""OpenFOAM result parser — reads logs and postProcessing data.

Parses:
- simpleFoam log files (residuals, convergence)
- forces/ directory (forces and moments on impeller)
- fieldAverage/ directory (mean pressure, velocity fields)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResidualData:
    """Residual convergence data from solver log."""

    iterations: list[int] = field(default_factory=list)
    p_residuals: list[float] = field(default_factory=list)
    ux_residuals: list[float] = field(default_factory=list)
    uy_residuals: list[float] = field(default_factory=list)
    uz_residuals: list[float] = field(default_factory=list)
    k_residuals: list[float] = field(default_factory=list)
    omega_residuals: list[float] = field(default_factory=list)
    converged: bool = False
    final_iteration: int = 0


@dataclass
class ForcesData:
    """Forces and moments on a patch from OpenFOAM forces function object."""

    time_steps: list[float] = field(default_factory=list)
    force_x: list[float] = field(default_factory=list)
    force_y: list[float] = field(default_factory=list)
    force_z: list[float] = field(default_factory=list)
    moment_x: list[float] = field(default_factory=list)
    moment_y: list[float] = field(default_factory=list)
    moment_z: list[float] = field(default_factory=list)


def parse_solver_log(log_path: Path | str) -> ResidualData:
    """Parse simpleFoam log file for residuals and convergence.

    Args:
        log_path: Path to the solver log file.

    Returns:
        ResidualData with iteration-by-iteration residuals.
    """
    log_path = Path(log_path)
    data = ResidualData()

    if not log_path.exists():
        return data

    content = log_path.read_text()

    # Pattern: "Time = 100"
    time_pattern = re.compile(r"^Time = (\d+)", re.MULTILINE)

    # Pattern: "Solving for Ux, Initial residual = 1.234e-05"
    residual_pattern = re.compile(
        r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)"
    )

    current_time = 0
    for line in content.split("\n"):
        time_match = time_pattern.match(line)
        if time_match:
            current_time = int(time_match.group(1))

        res_match = residual_pattern.search(line)
        if res_match:
            field_name = res_match.group(1)
            value = float(res_match.group(2))

            if field_name == "p":
                data.p_residuals.append(value)
                data.iterations.append(current_time)
            elif field_name == "Ux":
                data.ux_residuals.append(value)
            elif field_name == "Uy":
                data.uy_residuals.append(value)
            elif field_name == "Uz":
                data.uz_residuals.append(value)
            elif field_name == "k":
                data.k_residuals.append(value)
            elif field_name == "omega":
                data.omega_residuals.append(value)

    # Check convergence
    if data.p_residuals:
        data.final_iteration = current_time
        data.converged = data.p_residuals[-1] < 1e-4

    return data


def parse_forces(
    forces_dir: Path | str,
) -> ForcesData:
    """Parse forces postProcessing output.

    OpenFOAM writes forces to:
        postProcessing/forces1/<time>/force.dat

    Format: time ((fx fy fz) (fpx fpy fpz) (fvx fvy fvz))

    Args:
        forces_dir: Path to the forces function object directory.

    Returns:
        ForcesData with time-series of forces and moments.
    """
    forces_dir = Path(forces_dir)
    data = ForcesData()

    if not forces_dir.exists():
        return data

    # Find the latest time directory
    time_dirs = sorted(
        [d for d in forces_dir.iterdir() if d.is_dir()],
        key=lambda d: float(d.name) if d.name.replace(".", "").isdigit() else 0,
    )

    if not time_dirs:
        return data

    # Parse force.dat from latest time directory
    force_file = time_dirs[-1] / "force.dat"
    if not force_file.exists():
        # Try forces.dat (different OpenFOAM versions)
        force_file = time_dirs[-1] / "forces.dat"

    if not force_file.exists():
        return data

    # Parse the file
    number_pattern = re.compile(r"[0-9.eE+-]+")

    for line in force_file.read_text().split("\n"):
        if line.startswith("#") or not line.strip():
            continue

        numbers = [float(x) for x in number_pattern.findall(line)]
        if len(numbers) >= 7:
            data.time_steps.append(numbers[0])
            # Total force = pressure + viscous
            data.force_x.append(numbers[1])
            data.force_y.append(numbers[2])
            data.force_z.append(numbers[3])
        if len(numbers) >= 13:
            data.moment_x.append(numbers[7])
            data.moment_y.append(numbers[8])
            data.moment_z.append(numbers[9])

    return data
