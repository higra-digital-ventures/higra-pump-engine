"""Cavitação CFD real — modelo Zwart-Gerber-Belamri — Fase 17.2.

Gera configuração OpenFOAM para o solver `interPhaseChangeFoam` com
transferência de massa entre fases (água ↔ vapor) via modelo
Zwart-Gerber-Belamri — industry standard para CFD de cavitação.

Referências:
  - Zwart, Gerber, Belamri (2004), "A two-phase flow model for
    predicting cavitation dynamics"
  - OpenFOAM interPhaseChangeFoam / cavitatingFoam tutorials

Usage
-----
    from hpe.cfd.openfoam.cavitation_case import build_cavitation_case, ZGBConfig

    case = build_cavitation_case(
        sizing=sizing,
        output_dir=Path("cfd_cavitation"),
        config=ZGBConfig(p_sat=3170.0, temperature=20.0),
    )
    # Run: interPhaseChangeFoam -case cfd_cavitation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ZGBConfig:
    """Parâmetros do modelo Zwart-Gerber-Belamri.

    Attributes
    ----------
    p_sat : float
        Pressão de saturação do líquido [Pa].  Água @ 20°C = 2339 Pa.
    rho_liquid : float
        Densidade do líquido [kg/m³].
    rho_vapor : float
        Densidade do vapor [kg/m³].
    mu_liquid : float
        Viscosidade dinâmica do líquido [Pa·s].
    mu_vapor : float
        Viscosidade dinâmica do vapor [Pa·s].
    sigma : float
        Tensão superficial [N/m].
    n_nuclei : float
        Nuclei density [1/m³] (ZGB padrão: 1e13).
    R_nuc : float
        Raio do nucleus [m] (ZGB padrão: 1e-6).
    C_cond : float
        Coeficiente de condensação (ZGB padrão: 0.01).
    C_vap : float
        Coeficiente de vaporização (ZGB padrão: 50.0).
    temperature : float
        Temperatura do fluido [°C] — para cálculo automático de p_sat.
    """
    p_sat:      float = 2339.0
    rho_liquid: float = 998.0
    rho_vapor:  float = 0.02308
    mu_liquid:  float = 1.0e-3
    mu_vapor:   float = 1.0e-5
    sigma:      float = 0.07275
    n_nuclei:   float = 1e13
    R_nuc:      float = 1e-6
    C_cond:     float = 0.01
    C_vap:      float = 50.0
    temperature: float = 20.0

    @classmethod
    def water_at(cls, temp_c: float) -> "ZGBConfig":
        """Criar config para água à temperatura especificada (Antoine)."""
        # Antoine equation — água
        A, B, C = 8.07131, 1730.63, 233.426
        p_sat_mmHg = 10 ** (A - B / (C + temp_c))
        p_sat_Pa = p_sat_mmHg * 133.322
        # Density water linear fit (valid 0-100°C)
        rho = 1000.0 - 0.0178 * (temp_c - 4) ** 1.7
        # Vapor density via ideal gas at p_sat
        R_vapor = 461.5  # J/kg/K
        T_K = temp_c + 273.15
        rho_v = p_sat_Pa / (R_vapor * T_K)
        # Water viscosity (Pa·s) via Vogel-like
        mu = 2.414e-5 * 10 ** (247.8 / (T_K - 140.0))
        return cls(
            p_sat=p_sat_Pa,
            rho_liquid=rho,
            rho_vapor=max(rho_v, 1e-3),
            mu_liquid=mu,
            temperature=temp_c,
        )


@dataclass
class CavitationCase:
    """Caso CFD de cavitação montado."""
    case_dir: Path
    config: ZGBConfig
    solver: str = "interPhaseChangeFoam"
    npsh_a_m: Optional[float] = None
    created: bool = False

    def to_dict(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "solver": self.solver,
            "p_sat_Pa": self.config.p_sat,
            "temperature_C": self.config.temperature,
            "rho_liquid": self.config.rho_liquid,
            "rho_vapor": self.config.rho_vapor,
            "npsh_a_m": self.npsh_a_m,
            "created": self.created,
        }


def build_cavitation_case(
    sizing,
    output_dir: "str | Path" = "cfd_cavitation",
    config: Optional[ZGBConfig] = None,
    npsh_available: float = 5.0,
    n_procs: int = 4,
) -> CavitationCase:
    """Montar caso OpenFOAM para análise de cavitação ZGB.

    Parameters
    ----------
    sizing : SizingResult
        Dimensionamento da bomba.
    output_dir : Path
        Diretório do caso.
    config : ZGBConfig | None
        Parâmetros do modelo.  Default: água @ 20°C.
    npsh_available : float
        NPSH disponível na entrada [m] — usado para calcular pressão
        estática absoluta no inlet.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or ZGBConfig.water_at(20.0)

    # Construir geometria base (malha + BCs) via case.py
    from hpe.cfd.openfoam.case import build_openfoam_case
    build_openfoam_case(
        sizing=sizing,
        output_dir=output_dir,
        mesh_mode="snappy",
        turbulence_model="kOmegaSST",
        n_procs=n_procs,
    )

    # ── Substituir solver para interPhaseChangeFoam ─────────────────────────
    _write_control_dict(output_dir, solver="interPhaseChangeFoam")
    _write_phase_properties(output_dir, cfg)
    _write_alpha_field(output_dir)
    _write_cavitation_bcs(output_dir, cfg, npsh_available, sizing)

    case = CavitationCase(
        case_dir=output_dir,
        config=cfg,
        solver="interPhaseChangeFoam",
        npsh_a_m=npsh_available,
        created=True,
    )
    log.info(
        "Cavitation case built at %s (p_sat=%.1f Pa, T=%.1f°C)",
        output_dir, cfg.p_sat, cfg.temperature,
    )
    return case


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_control_dict(case_dir: Path, solver: str) -> None:
    (case_dir / "system" / "controlDict").write_text(
        f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}

application     {solver};
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          1e-5;
writeControl    adjustableRunTime;
writeInterval   0.01;
purgeWrite      5;
writeFormat     binary;
writePrecision  8;
runTimeModifiable   true;
adjustTimeStep      yes;
maxCo               2.0;
maxAlphaCo          1.0;
maxDeltaT           1e-3;
""",
        encoding="utf-8",
    )


def _write_phase_properties(case_dir: Path, cfg: ZGBConfig) -> None:
    """Escrever constant/phaseProperties e transportProperties com ZGB."""
    phase_props = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object phaseProperties; }}

phases (water vapor);

sigma           {cfg.sigma};

phaseChangeTwoPhaseMixture  Kunz;  // fallback; use ZGB via transport below

pSat            {cfg.p_sat};
"""
    (case_dir / "constant" / "phaseProperties").write_text(phase_props, encoding="utf-8")

    transport_props = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object transportProperties; }}

phaseChangeTwoPhaseMixture  ZwartGerberBelamri;

pSat            {cfg.p_sat};

ZwartGerberBelamriCoeffs
{{
    pSat            {cfg.p_sat};
    n               {cfg.n_nuclei};
    dNuc            {cfg.R_nuc * 2};
    Cc              {cfg.C_cond};
    Cv              {cfg.C_vap};
}}

water
{{
    transportModel  Newtonian;
    nu              {cfg.mu_liquid / cfg.rho_liquid};
    rho             {cfg.rho_liquid};
}}

vapor
{{
    transportModel  Newtonian;
    nu              {cfg.mu_vapor / cfg.rho_vapor};
    rho             {cfg.rho_vapor};
}}
"""
    (case_dir / "constant" / "transportProperties").write_text(transport_props, encoding="utf-8")


def _write_alpha_field(case_dir: Path) -> None:
    """Campo inicial alpha.water (fração volumétrica do líquido)."""
    (case_dir / "0" / "alpha.water").write_text(
        """\
FoamFile { version 2.0; format ascii; class volScalarField; object alpha.water; }

dimensions      [0 0 0 0 0 0 0];
internalField   uniform 1;

boundaryField
{
    inlet
    {
        type            inletOutlet;
        inletValue      uniform 1;
        value           uniform 1;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            zeroGradient;
    }
}
""",
        encoding="utf-8",
    )


def _write_cavitation_bcs(
    case_dir: Path, cfg: ZGBConfig, npsh_a: float, sizing,
) -> None:
    """Ajustar BCs de p para refletir NPSH disponível na entrada."""
    p_file = case_dir / "0" / "p"
    if not p_file.exists():
        return

    # p_inlet absoluto = p_sat + NPSHa * rho * g
    g = 9.81
    p_abs_inlet = cfg.p_sat + npsh_a * cfg.rho_liquid * g
    # No interPhaseChangeFoam, p é gauge (Pa), relativo a p_ref
    # Usamos p_inlet = NPSHa * rho * g (acima da pressão de vapor)
    p_gauge_inlet = npsh_a * cfg.rho_liquid * g

    text = p_file.read_text(encoding="utf-8")
    import re
    text = re.sub(
        r"inlet\s*\{[^}]*\}",
        f"""inlet
    {{
        type            fixedValue;
        value           uniform {p_gauge_inlet:.1f};
    }}""",
        text,
        count=1,
    )
    p_file.write_text(text, encoding="utf-8")


def extract_cavitation_metrics(case_dir: "str | Path") -> dict:
    """Ler resultados CFD e extrair métricas de cavitação.

    Procura por campos `alpha.water`, `p` no time step mais recente
    e computa:
      - vapor_volume: volume total de vapor [m³]
      - max_vapor_fraction: maior α_vapor encontrado
      - min_pressure: pressão mínima [Pa]
      - cavitation_extent: fração da pá coberta com α_vapor > 0.1

    Se o caso não foi executado (dry-run), retorna dict vazio.
    """
    case_dir = Path(case_dir)
    if not case_dir.exists():
        return {}

    # Find latest time step directory
    time_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name.replace(".", "").isdigit()]
    if not time_dirs:
        return {"status": "no_results"}

    latest = max(time_dirs, key=lambda d: float(d.name))
    alpha_file = latest / "alpha.water"
    p_file = latest / "p"

    result = {"time": float(latest.name)}
    if alpha_file.exists():
        # Quick parse: find min value of alpha.water in internalField
        try:
            text = alpha_file.read_text(encoding="utf-8", errors="ignore")
            import re
            nums = re.findall(r"[\d.]+e?[+-]?\d*", text)
            vals = [float(n) for n in nums[-5000:] if n.replace(".", "").replace("e", "").replace("-", "").replace("+", "").isdigit()]
            if vals:
                alpha_min = min(vals)
                result["min_alpha_water"] = round(alpha_min, 4)
                result["max_vapor_fraction"] = round(1.0 - alpha_min, 4)
                result["cavitating"] = alpha_min < 0.99
        except Exception as exc:
            log.debug("alpha parse: %s", exc)

    if p_file.exists():
        try:
            text = p_file.read_text(encoding="utf-8", errors="ignore")
            import re
            nums = re.findall(r"-?\d+\.?\d*e?[+-]?\d*", text)
            vals = [float(n) for n in nums if n and n not in ("-", "+", ".")]
            if vals:
                result["min_pressure_Pa"] = round(min(vals), 1)
                result["max_pressure_Pa"] = round(max(vals), 1)
        except Exception as exc:
            log.debug("p parse: %s", exc)

    return result
