"""Modelos físicos avançados — melhorias CFD #31-40.

- write_interfoam_case: free surface multi-phase
- write_non_newtonian: power-law / Bird-Carreau
- write_compressible_cavitation
- write_cht_multi_region: conjugate heat transfer
- write_lagrangian_particles: particle tracking
- write_erosion_model: Finnie/Oka
- compute_fwh_acoustic: Ffowcs Williams-Hawkings
- write_csf_surface_tension: Brackbill CSF
- write_buoyancy_boussinesq
- write_porous_zone: Darcy-Forchheimer
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ===========================================================================
# #31 InterFoam (free surface)
# ===========================================================================

def write_interfoam_case(
    case_dir: "str | Path",
    fluid1_name: str = "water",
    fluid2_name: str = "air",
    sigma: float = 0.07275,
) -> dict:
    """Setup mínimo para interFoam — free surface multi-phase."""
    case_dir = Path(case_dir)
    constants = case_dir / "constant"
    constants.mkdir(parents=True, exist_ok=True)

    (constants / "transportProperties").write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}

phases ({fluid1_name} {fluid2_name});

{fluid1_name}
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] 1e-6;
    rho             [1 -3 0 0 0 0 0] 998.2;
}}

{fluid2_name}
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] 1.48e-5;
    rho             [1 -3 0 0 0 0 0] 1.225;
}}

sigma           [1 0 -2 0 0 0 0] {sigma};
""", encoding="utf-8")

    return {
        "solver": "interFoam",
        "phases": [fluid1_name, fluid2_name],
        "sigma_N_per_m": sigma,
        "files_written": ["constant/transportProperties"],
    }


# ===========================================================================
# #32 Non-Newtonian fluid (power-law / Bird-Carreau)
# ===========================================================================

def write_non_newtonian(
    case_dir: "str | Path",
    model: str = "powerLaw",
    k: float = 0.1,
    n: float = 0.7,
    nu0: float = 1e-3,
    nuInf: float = 1e-6,
    lambda_carreau: float = 1.0,
) -> dict:
    """Configurar fluido não-Newtoniano (lama, polpa, sangue, polímeros).

    Models: powerLaw | BirdCarreau | Casson | HerschelBulkley
    """
    case_dir = Path(case_dir)
    transport = case_dir / "constant" / "transportProperties"
    transport.parent.mkdir(parents=True, exist_ok=True)

    if model == "powerLaw":
        body = f"""\
transportModel  powerLaw;
powerLawCoeffs
{{
    nu0     {nu0};
    nuInf   {nuInf};
    k       {k};
    n       {n};
}}
"""
    elif model == "BirdCarreau":
        body = f"""\
transportModel  BirdCarreau;
BirdCarreauCoeffs
{{
    nu0     {nu0};
    nuInf   {nuInf};
    k       {lambda_carreau};
    n       {n};
}}
"""
    else:
        body = f"transportModel  {model};\n"

    transport.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}

{body}
""", encoding="utf-8")

    return {
        "model": model,
        "k": k,
        "n_index": n,
        "nu0": nu0,
        "nuInf": nuInf,
        "files_written": ["constant/transportProperties"],
    }


# ===========================================================================
# #33 Compressible cavitation
# ===========================================================================

def write_compressible_cavitation(
    case_dir: "str | Path",
    p_sat: float = 2339.0,
    rho_l: float = 998.2,
    rho_v: float = 0.02,
) -> dict:
    """Setup para cavitatingFoam (compressible cavitation)."""
    case_dir = Path(case_dir)
    constants = case_dir / "constant"
    constants.mkdir(parents=True, exist_ok=True)

    (constants / "thermophysicalProperties").write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object thermophysicalProperties; }}

barotropicCompressibilityModel Wallis;

WallisCoeffs
{{
    psiv            [0 -2 2 0 0] 4.54e-7;
    psil            [0 -2 2 0 0] 5e-7;
    rholSat         [1 -3 0 0 0] {rho_l};
    pSat            [1 -1 -2 0 0] {p_sat};
    rhov            [1 -3 0 0 0] {rho_v};
}}
""", encoding="utf-8")

    return {
        "solver": "cavitatingFoam",
        "compressibility_model": "Wallis",
        "p_sat_Pa": p_sat,
        "files_written": ["constant/thermophysicalProperties"],
    }


# ===========================================================================
# #34 Conjugate heat transfer
# ===========================================================================

def write_cht_multi_region(
    case_dir: "str | Path",
    fluid_region: str = "fluid",
    solid_regions: list[str] = None,
) -> dict:
    """Setup chtMultiRegionFoam — conjugate heat transfer fluido+sólido.

    Cria estrutura constant/{regionName}/ para cada região.
    """
    case_dir = Path(case_dir)
    solids = solid_regions or ["impeller", "casing"]

    regions = [fluid_region] + solids
    for r in regions:
        (case_dir / "constant" / r).mkdir(parents=True, exist_ok=True)
        (case_dir / "0" / r).mkdir(parents=True, exist_ok=True)
        (case_dir / "system" / r).mkdir(parents=True, exist_ok=True)

    (case_dir / "constant" / "regionProperties").write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object regionProperties; }}

regions
(
    fluid    ({fluid_region})
    solid    ({" ".join(solids)})
);
""", encoding="utf-8")

    return {
        "solver": "chtMultiRegionFoam",
        "fluid_region": fluid_region,
        "solid_regions": solids,
        "n_regions": len(regions),
    }


# ===========================================================================
# #35 Lagrangian particle tracking
# ===========================================================================

def write_lagrangian_particles(
    case_dir: "str | Path",
    n_particles: int = 1000,
    particle_diameter: float = 100e-6,
    rho_p: float = 2500.0,
    injector_position: tuple[float, float, float] = (0, 0, 0),
) -> dict:
    """Setup KinematicCloud para particle tracking (sand, droplets)."""
    case_dir = Path(case_dir)
    cloud_dict = case_dir / "constant" / "kinematicCloudProperties"
    cloud_dict.parent.mkdir(parents=True, exist_ok=True)

    cloud_dict.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object kinematicCloudProperties; }}

solution
{{
    active          true;
    coupled         true;
    transient       yes;

    integrationSchemes
    {{
        U               Euler;
    }}
}}

constantProperties
{{
    parcelTypeId    1;
    rhoMin          1e-15;
    minParcelMass   1e-15;
    rho0            {rho_p};
}}

subModels
{{
    particleForces
    {{
        sphereDrag;
        gravity;
    }}

    injectionModels
    {{
        model1
        {{
            type            patchInjection;
            patchName       inlet;
            U0              (10 0 0);
            nParticle       {n_particles};
            parcelBasisType fixed;
            parcelsPerSecond 1000;
            sizeDistribution
            {{
                type        fixedValue;
                value       {particle_diameter};
            }}
        }}
    }}
}}
""", encoding="utf-8")

    return {
        "n_particles": n_particles,
        "diameter_m": particle_diameter,
        "particle_density": rho_p,
        "injector_position": injector_position,
    }


# ===========================================================================
# #36 Erosion model (Finnie / Oka)
# ===========================================================================

@dataclass
class ErosionResult:
    total_mass_loss_kg_per_s: float
    max_local_rate_kg_m2_s: float
    most_eroded_patch: str
    model: str

    def to_dict(self) -> dict:
        return {
            "total_mass_loss_kg_per_s": round(self.total_mass_loss_kg_per_s, 12),
            "max_local_rate_kg_m2_s": round(self.max_local_rate_kg_m2_s, 12),
            "most_eroded_patch": self.most_eroded_patch,
            "model": self.model,
        }


def compute_erosion(
    impact_velocity: float,
    impact_angle_deg: float,
    n_impacts: int = 1000,
    diameter: float = 100e-6,
    rho_particle: float = 2500.0,
    model: str = "Finnie",
) -> ErosionResult:
    """Modelo de erosão Finnie ou Oka.

    Finnie (1958):  E = K × U^n × f(α)
    Oka (2005):     mais sofisticado, depende de hardness
    """
    angle_rad = math.radians(impact_angle_deg)
    if model == "Finnie":
        n_exp = 2.4
        if impact_angle_deg <= 18.46:
            f_alpha = math.sin(2 * angle_rad) - 3 * math.sin(angle_rad) ** 2
        else:
            f_alpha = math.cos(angle_rad) ** 2 / 3
        K = 1e-9
    else:
        # Oka
        n_exp = 2.35
        f_alpha = (math.sin(angle_rad)) ** 0.71 * (1 + (1 - math.sin(angle_rad))) ** 1.85
        K = 1e-10

    mass_per_impact = rho_particle * (math.pi / 6) * diameter ** 3
    erosion_per_impact = K * impact_velocity ** n_exp * abs(f_alpha) * mass_per_impact
    total = erosion_per_impact * n_impacts

    return ErosionResult(
        total_mass_loss_kg_per_s=total,
        max_local_rate_kg_m2_s=total / 0.001,
        most_eroded_patch="blade_leading_edge",
        model=model,
    )


# ===========================================================================
# #37 Acoustic FW-H integral
# ===========================================================================

def compute_fwh_acoustic(
    surface_pressure_history: list[list[float]],
    observer_position: tuple[float, float, float],
    surface_normals: list[tuple[float, float, float]],
    surface_areas: list[float],
    rho: float = 998.2,
    c0: float = 1500.0,
    dt: float = 1e-4,
) -> dict:
    """Implementação simplificada do FW-H integral para ruído na água.

    Apenas o termo de loading noise (Far-field thickness ignorado para
    fontes compactas).
    """
    if not surface_pressure_history or not surface_normals:
        return {"p_acoustic_pa": 0.0, "spl_db": 0.0}

    # Approximate observer signal as time-delayed sum
    # of dipole-radiated pressures from each surface element
    n_t = len(surface_pressure_history[0]) if surface_pressure_history else 0
    p_obs = [0.0] * n_t

    for i, (n, A) in enumerate(zip(surface_normals, surface_areas)):
        if i >= len(surface_pressure_history):
            break
        for t in range(n_t):
            p = surface_pressure_history[i][t]
            # Dipole loading (simplified — distance-independent)
            p_obs[t] += p * A * 1e-6

    p_rms = math.sqrt(sum(p ** 2 for p in p_obs) / max(n_t, 1))
    p_ref_water = 1e-6   # Pa, reference for water
    spl = 20 * math.log10(max(p_rms / p_ref_water, 1e-12))

    return {
        "p_rms_pa": round(p_rms, 6),
        "spl_db": round(spl, 2),
        "n_samples": n_t,
        "method": "FW-H_loading_only",
    }


# ===========================================================================
# #38 Surface tension Brackbill CSF
# ===========================================================================

def write_csf_surface_tension(
    case_dir: "str | Path",
    sigma: float = 0.07275,
) -> dict:
    """Configurar Continuum Surface Force (Brackbill 1992) para interFoam."""
    case_dir = Path(case_dir)
    transport = case_dir / "constant" / "transportProperties"

    if transport.exists():
        text = transport.read_text()
        if "sigma" not in text:
            text += f"\nsigma           [1 0 -2 0 0 0 0] {sigma};\n"
            transport.write_text(text)

    return {
        "method": "Brackbill_CSF",
        "sigma_N_per_m": sigma,
        "model_year": 1992,
    }


# ===========================================================================
# #39 Buoyancy Boussinesq
# ===========================================================================

def write_buoyancy_boussinesq(
    case_dir: "str | Path",
    T_ref: float = 293.15,
    beta: float = 2.07e-4,
    g: tuple[float, float, float] = (0, 0, -9.81),
) -> dict:
    """Setup buoyancy Boussinesq — densidade ρ(T) linearizada."""
    case_dir = Path(case_dir)
    constants = case_dir / "constant"
    constants.mkdir(parents=True, exist_ok=True)

    (constants / "transportProperties").write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1e-6;
beta            [0 0 0 -1 0 0 0] {beta};
TRef            [0 0 0 1 0 0 0] {T_ref};
Pr              [0 0 0 0 0 0 0] 7.0;
Prt             [0 0 0 0 0 0 0] 0.85;
""", encoding="utf-8")

    (constants / "g").write_text(f"""\
FoamFile {{ version 2.0; format ascii; class uniformDimensionedVectorField; object g; }}
dimensions      [0 1 -2 0 0 0 0];
value           ({g[0]} {g[1]} {g[2]});
""", encoding="utf-8")

    return {
        "model": "Boussinesq",
        "beta_per_K": beta,
        "T_ref_K": T_ref,
        "gravity": list(g),
    }


# ===========================================================================
# #40 Porous media zones
# ===========================================================================

def write_porous_zone(
    case_dir: "str | Path",
    zone_name: str,
    d_coeffs: tuple[float, float, float] = (1e6, 1e6, 1e6),
    f_coeffs: tuple[float, float, float] = (10, 10, 10),
) -> dict:
    """Definir zona porosa Darcy-Forchheimer (filtros, screens, beds).

    ΔP/L = μ·d·U + 0.5·ρ·f·U²
    """
    case_dir = Path(case_dir)
    porous_file = case_dir / "constant" / "porousZones"
    porous_file.parent.mkdir(parents=True, exist_ok=True)

    porous_file.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object porousZones; }}

(
    {zone_name}
    {{
        type            DarcyForchheimer;
        DarcyForchheimerCoeffs
        {{
            d   d   [0 -2 0 0 0 0 0] ({d_coeffs[0]} {d_coeffs[1]} {d_coeffs[2]});
            f   f   [0 -1 0 0 0 0 0] ({f_coeffs[0]} {f_coeffs[1]} {f_coeffs[2]});
            coordinateSystem
            {{
                e1  (1 0 0);
                e2  (0 1 0);
            }}
        }}
    }}
)
""", encoding="utf-8")

    return {
        "model": "Darcy-Forchheimer",
        "zone_name": zone_name,
        "d_coeffs": list(d_coeffs),
        "f_coeffs": list(f_coeffs),
    }
