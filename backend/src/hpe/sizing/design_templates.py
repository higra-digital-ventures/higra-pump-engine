"""Pre-loaded design templates for turbomachinery sizing.

Provides 12 complete example designs with real-world parameters spanning
centrifugal pumps, turbines, fans, compressors and pump-turbines.
"""

from __future__ import annotations

from typing import Any, Dict, List

from hpe.core.enums import MachineType
from hpe.core.models import OperatingPoint


TEMPLATES: Dict[str, Dict[str, Any]] = {
    "centrifugal_pump_low_nq": {
        "name": "Bomba Centrífuga Baixo Nq (Nq=18)",
        "description": "Bomba de alta pressão, baixa vazão — típica de água de alimentação de caldeira",
        "flow_rate_m3h": 50,
        "head_m": 80,
        "rpm": 3550,
        "machine_type": "centrifugal_pump",
        "fluid": "water",
        "expected_nq": 18,
        "expected_eta": 0.78,
        "expected_z": 7,
    },
    "centrifugal_pump_medium_nq": {
        "name": "Bomba Centrífuga Médio Nq (Nq=30)",
        "description": "Bomba industrial padrão — água de processo",
        "flow_rate_m3h": 100,
        "head_m": 32,
        "rpm": 1750,
        "machine_type": "centrifugal_pump",
        "fluid": "water",
        "expected_nq": 30,
        "expected_eta": 0.82,
        "expected_z": 6,
    },
    "centrifugal_pump_high_nq": {
        "name": "Bomba Centrífuga Alto Nq (Nq=80)",
        "description": "Bomba de grande vazão — irrigação, drenagem",
        "flow_rate_m3h": 1000,
        "head_m": 20,
        "rpm": 1750,
        "machine_type": "centrifugal_pump",
        "fluid": "water",
        "expected_nq": 80,
        "expected_eta": 0.87,
        "expected_z": 5,
    },
    "mixed_flow_pump": {
        "name": "Bomba Mixed-Flow (Nq=120)",
        "description": "Bomba mista — estação elevatória de esgoto",
        "flow_rate_m3h": 3000,
        "head_m": 12,
        "rpm": 980,
        "machine_type": "centrifugal_pump",
        "fluid": "water",
        "expected_nq": 120,
        "expected_eta": 0.88,
        "expected_z": 4,
    },
    "francis_turbine_medium": {
        "name": "Turbina Francis Média Queda",
        "description": "Turbina Francis PCH — queda 50m, 2MW",
        "flow_rate_m3h": 5000,
        "head_m": 50,
        "rpm": 600,
        "machine_type": "francis_turbine",
        "fluid": "water",
        "expected_nq": 65,
        "expected_eta": 0.91,
        "expected_z": 13,
    },
    "francis_turbine_high_head": {
        "name": "Turbina Francis Alta Queda",
        "description": "Turbina Francis grande porte — queda 200m",
        "flow_rate_m3h": 20000,
        "head_m": 200,
        "rpm": 375,
        "machine_type": "francis_turbine",
        "fluid": "water",
        "expected_nq": 40,
        "expected_eta": 0.93,
        "expected_z": 15,
    },
    "radial_turbine_orc": {
        "name": "Turbina Radial ORC (R134a)",
        "description": "Turbina radial para ciclo ORC — recuperação de calor",
        "flow_rate_m3h": 10,
        "head_m": 25,
        "rpm": 12000,
        "machine_type": "radial_turbine",
        "fluid": "R134A",
        "expected_nq": 45,
        "expected_eta": 0.82,
        "expected_z": 12,
    },
    "axial_fan_industrial": {
        "name": "Ventilador Axial Industrial",
        "description": "Ventilador de exaustão — fábrica, AVAC",
        "flow_rate_m3h": 50000,
        "head_m": 0.15,
        "rpm": 1450,
        "machine_type": "axial_fan",
        "fluid": "air",
        "expected_nq": 200,
        "expected_eta": 0.78,
        "expected_z": 8,
    },
    "centrifugal_compressor": {
        "name": "Compressor Centrífugo (Ar)",
        "description": "Compressor de ar industrial — pressão 3:1",
        "flow_rate_m3h": 5000,
        "head_m": 8000,
        "rpm": 15000,
        "machine_type": "centrifugal_pump",
        "fluid": "air",
        "expected_nq": 25,
        "expected_eta": 0.80,
        "expected_z": 17,
    },
    "sirocco_fan_hvac": {
        "name": "Ventilador Sirocco AVAC",
        "description": "Ventilador FC para ar condicionado",
        "flow_rate_m3h": 3000,
        "head_m": 0.05,
        "rpm": 800,
        "machine_type": "sirocco_fan",
        "fluid": "air",
        "expected_nq": 300,
        "expected_eta": 0.55,
        "expected_z": 36,
    },
    "multistage_boiler_feed": {
        "name": "Bomba Multistágio Alimentação Caldeira",
        "description": "5 estágios, alta pressão — 200 bar",
        "flow_rate_m3h": 80,
        "head_m": 500,
        "rpm": 3550,
        "machine_type": "centrifugal_pump",
        "fluid": "water",
        "expected_nq": 22,
        "expected_eta": 0.75,
        "n_stages": 5,
    },
    "pump_turbine_reversible": {
        "name": "Pump-Turbine Reversível",
        "description": "Armazenamento por bombeamento — ciclo pump/turbine",
        "flow_rate_m3h": 15000,
        "head_m": 300,
        "rpm": 428,
        "machine_type": "francis_turbine",
        "fluid": "water",
        "expected_nq": 35,
        "expected_eta": 0.90,
        "expected_z": 7,
    },
}


def get_template(name: str) -> Dict[str, Any]:
    """Return a single template by key name.

    Args:
        name: Template key (e.g., 'centrifugal_pump_low_nq').

    Returns:
        Full template dict including sizing parameters and expected values.

    Raises:
        KeyError: If template name does not exist.
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return {**TEMPLATES[name], "key": name}


def list_templates() -> List[Dict[str, Any]]:
    """Return summary list of all templates.

    Returns:
        List of dicts with key, name, description, machine_type, expected_nq.
    """
    result = []
    for key, tpl in TEMPLATES.items():
        result.append({
            "key": key,
            "name": tpl["name"],
            "description": tpl["description"],
            "machine_type": tpl["machine_type"],
            "expected_nq": tpl.get("expected_nq"),
            "expected_eta": tpl.get("expected_eta"),
            "fluid": tpl.get("fluid", "water"),
        })
    return result


def run_template(name: str) -> Dict[str, Any]:
    """Run 1D sizing using a template's parameters.

    Args:
        name: Template key.

    Returns:
        Dict with template metadata and full sizing result.

    Raises:
        KeyError: If template name does not exist.
    """
    from hpe.sizing import run_sizing

    tpl = get_template(name)

    # Convert m3/h to m3/s
    flow_rate_m3s = tpl["flow_rate_m3h"] / 3600.0

    op = OperatingPoint(
        flow_rate=flow_rate_m3s,
        head=tpl["head_m"],
        rpm=tpl["rpm"],
        machine_type=MachineType(tpl["machine_type"]),
    )

    result = run_sizing(op)

    uncertainty = result.uncertainty.as_dict() if result.uncertainty else {}

    return {
        "template": tpl,
        "sizing": {
            "specific_speed_nq": result.specific_speed_nq,
            "impeller_type": result.meridional_profile.get("impeller_type", "unknown"),
            "impeller_d2": result.impeller_d2,
            "impeller_d1": result.impeller_d1,
            "impeller_b2": result.impeller_b2,
            "blade_count": result.blade_count,
            "beta1": result.beta1,
            "beta2": result.beta2,
            "estimated_efficiency": result.estimated_efficiency,
            "estimated_power": result.estimated_power,
            "estimated_npsh_r": result.estimated_npsh_r,
            "sigma": result.sigma,
            "velocity_triangles": result.velocity_triangles,
            "meridional_profile": result.meridional_profile,
            "warnings": result.warnings,
            "uncertainty": uncertainty,
        },
    }
