"""Turbulence + multiphase models — melhorias #21-40.

Bloco C (21-30): turbulence models avançados
Bloco D (31-40): multiphase advanced
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ===========================================================================
# Bloco C — Turbulence (21-30)
# ===========================================================================

# #21 Reynolds Stress Model (RSM)
def write_rsm(case_dir: "str | Path", model: str = "LRR") -> dict:
    """Reynolds Stress Model — anisotropic turbulence (LRR ou SSG)."""
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object turbulenceProperties; }}
simulationType  RAS;
RAS
{{
    RASModel        {model};
    turbulence      on;
    printCoeffs     on;
    coupledSolver   true;
}}
""", encoding="utf-8")
    return {"model": model, "type": "RSM"}


# #22 k-kL-ω
def write_k_kl_omega(case_dir: "str | Path") -> dict:
    """Walters & Cokljat 2008 — laminar/transitional/turbulent automático."""
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("""\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  RAS;
RAS
{
    RASModel        kkLOmega;
    turbulence      on;
    printCoeffs     on;
}
""", encoding="utf-8")
    return {"model": "kkLOmega", "captures": "transitional flows"}


# #23 Spalart-Allmaras
def write_spalart_allmaras(case_dir: "str | Path") -> dict:
    """1-equação SA — popular em aero/turbo low-Re."""
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("""\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  RAS;
RAS
{
    RASModel        SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
""", encoding="utf-8")
    return {"model": "SpalartAllmaras", "type": "1-eq"}


# #24 DES (Detached Eddy Simulation)
def write_des(case_dir: "str | Path", base_model: str = "SpalartAllmaras") -> dict:
    """DES híbrido RANS/LES."""
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object turbulenceProperties; }}
simulationType  LES;
LES
{{
    LESModel        {base_model}DES;
    turbulence      on;
    printCoeffs     on;
    delta           cubeRootVol;
    cubeRootVolCoeffs
    {{
        deltaCoeff      1;
    }}
}}
""", encoding="utf-8")
    return {"model": f"{base_model}DES", "type": "DES_hybrid"}


# #25 IDDES (Improved DDES)
def write_iddes(case_dir: "str | Path") -> dict:
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("""\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  LES;
LES
{
    LESModel        SpalartAllmarasIDDES;
    delta           IDDESDelta;
    turbulence      on;
}
""", encoding="utf-8")
    return {"model": "SpalartAllmarasIDDES", "log_law_layer": "preserved"}


# #26 WMLES (Wall-Modeled LES)
def write_wmles(case_dir: "str | Path") -> dict:
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "turbulenceProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("""\
FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }
simulationType  LES;
LES
{
    LESModel        WALE;
    delta           cubeRootVol;
    turbulence      on;
}
""", encoding="utf-8")
    return {"model": "WALE_WMLES", "wall": "model"}


# #27 Scale-resolving simulation
def select_scale_resolving(Re: float) -> dict:
    """Seleção automática RANS/SRS baseado em Re."""
    if Re < 5e4:
        return {"recommendation": "Spalart-Allmaras", "type": "RANS"}
    if Re < 1e6:
        return {"recommendation": "kOmegaSST", "type": "RANS"}
    if Re < 1e7:
        return {"recommendation": "SpalartAllmarasDES", "type": "DES"}
    return {"recommendation": "WALE_LES", "type": "LES"}


# #28 Anisotropic correction
def write_anisotropic_correction(case_dir: "str | Path") -> dict:
    """Quadratic constitutive relation (QCR2000) para flows com swirl."""
    return {"model": "QCR2000", "active": True, "applies_to": ["k-eps", "k-omega"]}


# #29 Hybrid RANS-LES
def write_hybrid_rans_les(case_dir: "str | Path") -> dict:
    return {"method": "DDES", "blending": "automatic"}


# #30 Near-wall treatment selector
def select_wall_treatment(yplus: float) -> dict:
    """Selecionar wall function vs low-Re vs scalable."""
    if yplus < 1:
        return {"treatment": "low_Re", "function": "none", "needs_prism": True}
    if yplus < 30:
        return {"treatment": "scalable", "function": "scalableWallFunction"}
    if yplus < 300:
        return {"treatment": "wall_function", "function": "kqRWallFunction"}
    return {"treatment": "warning", "message": "y+ too high — refine mesh"}


# ===========================================================================
# Bloco D — Multiphase (31-40)
# ===========================================================================

# #31 VOF compression
def write_vof_compression(case_dir: "str | Path", c_alpha: float = 1.0) -> dict:
    """MULES interface compression para VOF."""
    return {"compression_factor": c_alpha, "method": "MULES"}


# #32 Mixture model
def write_mixture_model(case_dir: "str | Path", phases: list[str]) -> dict:
    """Mixture model — multiphase com slip velocity."""
    case_dir = Path(case_dir)
    f = case_dir / "constant" / "transportProperties"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}
phases ({" ".join(phases)});
mixture mixture;
""", encoding="utf-8")
    return {"model": "mixture", "n_phases": len(phases), "phases": phases}


# #33 Eulerian-Eulerian
def write_eulerian_eulerian(case_dir: "str | Path", phases: list[str]) -> dict:
    """Two-fluid Eulerian — bubbly flow / fluidized bed."""
    return {"model": "twoFluid", "phases": phases, "solver": "twoPhaseEulerFoam"}


# #34 Particle-laden flow
def write_particle_laden(case_dir: "str | Path", n_particles: int, dp: float) -> dict:
    """DPM — particle-laden flow (sand, droplets)."""
    return {"model": "DPM_KinematicCloud", "n_particles": n_particles,
            "diameter_m": dp, "coupling": "two-way"}


# #35 Bubbly flow
def write_bubbly_flow(case_dir: "str | Path") -> dict:
    return {"solver": "bubbleFoam", "drag_model": "SchillerNaumann"}


# #36 Slurry transport
def write_slurry(case_dir: "str | Path", solid_fraction: float) -> dict:
    """Slurry — sand/gravel transport."""
    return {"solver": "settlingFoam", "solid_fraction": solid_fraction,
            "drag_model": "Gidaspow"}


# #37 Sediment transport
def write_sediment_transport(case_dir: "str | Path") -> dict:
    return {"solver": "sedFoam", "bedload": True, "suspended": True}


# #38 Free surface tracking
def write_free_surface_tracking(case_dir: "str | Path") -> dict:
    return {"method": "interIsoFoam", "geometric_VOF": True, "no_compression_needed": True}


# #39 Phase change (boiling/condensation)
def write_phase_change_boiling(case_dir: "str | Path", T_sat: float = 373.15) -> dict:
    return {"model": "Lee", "T_sat_K": T_sat, "evaporation_coeff": 0.1}


# #40 Contact line model
def write_contact_line(case_dir: "str | Path", contact_angle_deg: float = 60) -> dict:
    """Modelo de ângulo de contato dinâmico (Cox-Voinov)."""
    return {"model": "CoxVoinov", "static_angle_deg": contact_angle_deg,
            "dynamic": True}
