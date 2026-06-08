"""TURBOdesignPre .ptd file reader.

Parses the keyword-value format used by TURBOdesignPre to define
turbomachinery design projects. Converts PTD parameters to HPE
OperatingPoint and SizingRequest format.

PTD format:
    KEY = VALUE  (one per line, comments start with #)
    Sections are separated by ### SECTION NAME ### headers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


PTD_MACHINE_TYPES = {
    "0": "centrifugal_compressor",
    "1": "centrifugal_pump",
    "2": "radial_inflow_turbine",
    "3": "centrifugal_fan",
    "4": "mixed_flow_pump",
    "5": "axial_fan",
    "6": "axial_pump",
    "7": "refrigeration_compressor",
    "8": "francis_turbine",
    "9": "axial_compressor",
}


def parse_ptd(path: str | Path) -> dict[str, Any]:
    """Parse a .ptd file and return a dict of parameter key-value pairs.

    Args:
        path: Path to the .ptd file.

    Returns:
        Dict with all parameters. Nested structure:
        {
            "_machine_type_code": "1",
            "SPEC_VOLUME_FLOW_RATE": 0.107,
            "SPEC_PUMP_HEAD": 16.773,
            ...
        }
    """
    params: dict[str, Any] = {}
    current_section = ""

    with open(Path(path), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header
            if line.startswith("###"):
                current_section = line.strip("#").strip()
                continue

            # Key = Value
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                params[key] = _parse_value(value)

    return params


def ptd_to_operating_point(params: dict) -> dict:
    """Convert PTD parameters to HPE OperatingPoint dict.

    Returns a dict suitable for constructing OperatingPoint.
    """
    op = {}

    # Machine type
    mt_code = str(int(params.get("MACHINE_TYPE", params.get("SPEC_MACHINE_TYPE", 1))))
    op["machine_type"] = PTD_MACHINE_TYPES.get(mt_code, "centrifugal_pump")

    # Flow rate (m³/s)
    if "SPEC_VOLUME_FLOW_RATE" in params:
        op["flow_rate"] = float(params["SPEC_VOLUME_FLOW_RATE"])
    elif "SPEC_MASS_FLOW_RATE" in params:
        rho = float(params.get("SPEC_DENSITY", 998.0))
        op["flow_rate"] = float(params["SPEC_MASS_FLOW_RATE"]) / rho

    # Head (m)
    if "SPEC_PUMP_HEAD" in params:
        op["head"] = float(params["SPEC_PUMP_HEAD"])
    elif "SPEC_PRESSURE_RISE" in params:
        rho = float(params.get("SPEC_DENSITY", 998.0))
        op["head"] = float(params["SPEC_PRESSURE_RISE"]) / (rho * 9.81)

    # Speed
    if "SPEC_ROTATIONAL_SPEED" in params:
        op["rpm"] = float(params["SPEC_ROTATIONAL_SPEED"])

    # Fluid
    if "SPEC_DENSITY" in params:
        op["fluid_density"] = float(params["SPEC_DENSITY"])
    if "SPEC_VISCOSITY" in params:
        op["fluid_viscosity"] = float(params["SPEC_VISCOSITY"])
    if "SPEC_VAPOUR_PRESSURE" in params:
        op["vapor_pressure"] = float(params["SPEC_VAPOUR_PRESSURE"])

    # Convergence
    if "CONVERGENCE_INITIAL_STAGE_EFFICIENCY" in params:
        op["initial_efficiency_guess"] = float(params["CONVERGENCE_INITIAL_STAGE_EFFICIENCY"])

    return op


def _parse_value(value: str) -> Any:
    """Try to parse as float, int, or list of floats."""
    # Try list (space-separated numbers)
    parts = value.split()
    if len(parts) > 1:
        try:
            return [float(p) for p in parts]
        except ValueError:
            pass
    # Try single number
    try:
        return float(value)
    except ValueError:
        return value
