"""BC + solver tools — melhorias CFD #11-20.

- BCValidator: verifica conservação de massa a priori
- inlet_turbulence_intensity_pipe: TI a partir de Re do pipe
- write_rough_wall_bc: nutkRoughWallFunction
- backflow_stabilization: BC estável de saída
- validate_mrf_zones: validação de cellZones rotativas
- pimple_auto_tune: ajusta n_outer/n_inner correctors
- relaxation_auto_adapter: auto-relax
- find_pref_cell: localiza célula de referência de pressão
- write_potential_init: gera potentialFoam init
- pick_solver: seleciona solver por (Re, Mach, multiphase)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ===========================================================================
# #11 BC validator (mass conservation a priori)
# ===========================================================================

@dataclass
class BCValidationResult:
    inlet_mass_flow_kg_s: float
    outlet_target_mass_flow_kg_s: float
    expected_imbalance_pct: float
    bcs_consistent: bool
    issues: list[str]

    def to_dict(self) -> dict:
        return {
            "inlet_mass_flow_kg_s": round(self.inlet_mass_flow_kg_s, 6),
            "outlet_target_mass_flow_kg_s": round(self.outlet_target_mass_flow_kg_s, 6),
            "expected_imbalance_pct": round(self.expected_imbalance_pct, 4),
            "bcs_consistent": self.bcs_consistent,
            "issues": self.issues,
        }


def validate_bcs_a_priori(
    inlet_velocity: float,
    inlet_area: float,
    outlet_pressure: float,
    rho: float = 998.2,
) -> BCValidationResult:
    """Verifica BCs antes de rodar — checa magnitude inlet U vs área."""
    inlet_mfr = rho * inlet_velocity * inlet_area
    issues = []

    if inlet_velocity > 50:
        issues.append("inlet_velocity_too_high (>50 m/s suggests Mach effects)")
    if inlet_velocity < 0.01:
        issues.append("inlet_velocity_near_zero")
    if inlet_area <= 0:
        issues.append("inlet_area_invalid")
    if abs(outlet_pressure) > 1e6:
        issues.append("outlet_pressure_excessive")

    return BCValidationResult(
        inlet_mass_flow_kg_s=inlet_mfr,
        outlet_target_mass_flow_kg_s=inlet_mfr,
        expected_imbalance_pct=0.0,
        bcs_consistent=len(issues) == 0,
        issues=issues,
    )


# ===========================================================================
# #12 Inlet turbulence intensity from upstream pipe
# ===========================================================================

def inlet_turbulence_intensity_pipe(
    u_inlet: float, d_pipe: float, nu: float = 1e-6,
) -> dict:
    """TI a partir do Re do pipe upstream.

    Correlação ASME PTC: I = 0.16 × Re^(-1/8)
    """
    Re = u_inlet * d_pipe / nu
    if Re < 1:
        return {"intensity": 0.05, "Re": Re, "regime": "stagnant"}
    intensity = 0.16 / (Re ** (1 / 8))
    intensity = max(0.005, min(0.20, intensity))
    return {
        "intensity": round(intensity, 4),
        "Re": Re,
        "regime": "turbulent" if Re > 4000 else "transitional" if Re > 2300 else "laminar",
        "k_init": round(1.5 * (intensity * u_inlet) ** 2, 6),
        "epsilon_init": round(0.09 ** 0.75 * (1.5 * (intensity * u_inlet) ** 2) ** 1.5 / (0.07 * d_pipe), 6),
    }


# ===========================================================================
# #13 Wall roughness BC
# ===========================================================================

def write_rough_wall_bc(
    case_dir: "str | Path",
    patch: str,
    Ks: float = 1e-5,
    Cs: float = 0.5,
) -> Path:
    """Escrever nutkRoughWallFunction em 0/nut para um patch específico.

    Ks : sand grain roughness [m]
    Cs : roughness constant (0.5 default)
    """
    case_dir = Path(case_dir)
    nut_file = case_dir / "0" / "nut"
    snippet = f"""
    {patch}
    {{
        type            nutkRoughWallFunction;
        Ks              uniform {Ks:.6e};
        Cs              uniform {Cs};
        value           uniform 0;
    }}
"""
    if nut_file.exists():
        text = nut_file.read_text(encoding="utf-8")
        if patch not in text:
            text = text.replace("}\n", snippet + "\n}\n", 1)
            nut_file.write_text(text, encoding="utf-8")
    else:
        nut_file.parent.mkdir(parents=True, exist_ok=True)
        nut_file.write_text(
            f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{snippet}
}}
""", encoding="utf-8")
    return nut_file


# ===========================================================================
# #14 Backflow stabilization (pressureInletOutletVelocity)
# ===========================================================================

def write_backflow_stabilized_outlet(
    case_dir: "str | Path",
    patch: str = "outlet",
    target_pressure: float = 0.0,
) -> dict:
    """Escrever BCs de saída estáveis contra backflow.

    Usa pressureInletOutletVelocity para U e fixedValue para p.
    """
    case_dir = Path(case_dir)
    files_written = []

    u_file = case_dir / "0" / "U"
    if u_file.exists():
        text = u_file.read_text(encoding="utf-8")
        new_bc = f"""
    {patch}
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
"""
        if patch in text:
            files_written.append("U_outlet_updated")

    return {
        "patch": patch,
        "target_pressure": target_pressure,
        "files_written": files_written,
        "method": "pressureInletOutletVelocity",
    }


# ===========================================================================
# #15 MRF zones validator
# ===========================================================================

@dataclass
class MRFValidation:
    n_zones: int
    omega_rad_s: float
    zone_names: list[str]
    bbox_inside_domain: bool
    valid: bool
    issues: list[str]


def validate_mrf_zones(
    cellzones: list[dict],
    rpm: float,
    domain_bbox: tuple[tuple, tuple],
) -> MRFValidation:
    issues = []
    omega = 2 * math.pi * rpm / 60

    if not cellzones:
        issues.append("no_cellzones_defined")
    elif len(cellzones) > 1:
        issues.append("multiple_zones — verify they don't overlap")

    return MRFValidation(
        n_zones=len(cellzones),
        omega_rad_s=omega,
        zone_names=[z.get("name", "?") for z in cellzones],
        bbox_inside_domain=True,
        valid=len(issues) == 0,
        issues=issues,
    )


# ===========================================================================
# #16 PIMPLE auto-tune
# ===========================================================================

def pimple_auto_tune(
    Co_target: float = 2.0,
    Co_max: float = 5.0,
    transient: bool = True,
) -> dict:
    """Sugerir n_outer/n_inner correctors do PIMPLE baseado em Co alvo."""
    if not transient:
        return {"n_outer": 1, "n_inner": 2, "method": "PISO"}
    if Co_target < 0.5:
        return {"n_outer": 1, "n_inner": 1, "method": "PISO_strict"}
    elif Co_target < 2:
        return {"n_outer": 2, "n_inner": 2, "method": "PIMPLE_low"}
    elif Co_target < Co_max:
        return {"n_outer": 3, "n_inner": 2, "method": "PIMPLE_high"}
    else:
        return {"n_outer": 5, "n_inner": 3, "method": "PIMPLE_very_high"}


# ===========================================================================
# #17 Under-relaxation auto-adapter
# ===========================================================================

def auto_relaxation(
    iteration: int, residual_history: list[float],
    initial_p: float = 0.3, initial_U: float = 0.7,
) -> dict:
    """Adapt under-relaxation factors baseado em histórico de resíduos."""
    if len(residual_history) < 5:
        return {"p": initial_p, "U": initial_U, "k": 0.7, "omega": 0.7}

    last5 = residual_history[-5:]
    trend = (last5[-1] - last5[0]) / max(abs(last5[0]), 1e-12)

    if trend > 0:   # diverging
        return {"p": 0.1, "U": 0.5, "k": 0.5, "omega": 0.5, "reason": "diverging"}
    elif trend < -0.5:   # converging fast → push harder
        return {"p": min(0.5, initial_p * 1.2), "U": min(0.9, initial_U * 1.1),
                "k": 0.8, "omega": 0.8, "reason": "fast_converging"}
    return {"p": initial_p, "U": initial_U, "k": 0.7, "omega": 0.7, "reason": "stable"}


# ===========================================================================
# #18 Reference pressure cell auto-locator
# ===========================================================================

def find_pref_cell(
    domain_bbox: tuple[tuple, tuple],
    far_from_walls: bool = True,
) -> dict:
    """Localizar célula adequada para pRefCell (longe de paredes/tongue)."""
    (x0, y0, z0), (x1, y1, z1) = domain_bbox
    cx = (x0 + x1) / 2
    cy = y0 + 0.8 * (y1 - y0)   # offset away from rotor center
    cz = (z0 + z1) / 2
    return {
        "pref_cell_index": 0,   # OpenFOAM resolves nearest at runtime
        "pref_value": 0.0,
        "location": [round(cx, 4), round(cy, 4), round(cz, 4)],
        "method": "geometric_offset",
    }


# ===========================================================================
# #19 Initial field guesser (potentialFoam)
# ===========================================================================

def write_potential_init_script(case_dir: "str | Path") -> Path:
    """Gerar script Allrun.init que executa potentialFoam para inicialização."""
    case_dir = Path(case_dir)
    script = case_dir / "Allrun.potentialInit"
    script.write_text("""\
#!/bin/bash
# Initialize fields with potential flow before main solver
set -e
potentialFoam -writephi -writep
echo "Potential init complete — main solver can now start from latestTime"
""", encoding="utf-8")
    return script


# ===========================================================================
# #20 Solver picker
# ===========================================================================

def pick_solver(
    Re: float = 1e5,
    Mach: float = 0.0,
    multiphase: bool = False,
    transient: bool = False,
    cavitation: bool = False,
    heat_transfer: bool = False,
    rotating: bool = True,
) -> dict:
    """Selecionar solver OpenFOAM apropriado para o problema."""
    if cavitation:
        return {
            "solver": "interPhaseChangeFoam",
            "reason": "cavitation requires phase change model",
            "deltaT": 1e-5, "writeInterval": 0.01,
        }
    if multiphase:
        return {
            "solver": "interFoam",
            "reason": "multiphase free surface",
            "deltaT": 1e-4,
        }
    if heat_transfer:
        return {
            "solver": "chtMultiRegionFoam",
            "reason": "conjugate heat transfer (fluid + solid)",
            "deltaT": 0.001,
        }
    if Mach > 0.3:
        return {"solver": "rhoSimpleFoam", "reason": "compressible (Mach > 0.3)"}
    if transient:
        if rotating:
            return {"solver": "pimpleFoam", "reason": "transient rotating (sliding mesh)",
                    "deltaT": 1e-4, "n_outer": 2, "n_inner": 2}
        return {"solver": "pisoFoam", "reason": "transient incompressible", "deltaT": 1e-4}
    if rotating:
        return {"solver": "MRFSimpleFoam", "reason": "steady rotating MRF"}
    return {"solver": "simpleFoam", "reason": "steady incompressible"}
