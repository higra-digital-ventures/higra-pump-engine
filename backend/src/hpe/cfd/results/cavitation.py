"""Análise de cavitação — índice σ, NPSHr, NPSHa — Fase 13.

Implementa as correlações de Gülich (2014) §6.3–6.5 para:
  - NPSHr (required NPSH) via correlação de velocidade específica
  - Índice de Thoma σ = NPSHa / H
  - Verificação de margem de cavitação
  - Distribuição de pressão mínima na pá (risco de cavitação local)

Usage
-----
    from hpe.cfd.results.cavitation import assess_cavitation, CavitationAssessment

    result = assess_cavitation(sizing_result, npsh_available=5.0)
    print(result.safe)
    print(result.sigma_plant)
    print(result.npsh_r)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpe.core.models import OperatingPoint

log = logging.getLogger(__name__)

_RHO_WATER  = 998.2   # kg/m³ @ 20°C
_G          = 9.80665
_P_VAP_20C  = 2338.0  # Pa — pressão de vapor da água a 20°C


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CavitationAssessment:
    """Avaliação de risco de cavitação.

    Attributes
    ----------
    npsh_r : float
        NPSH requerido pela bomba [m] (Gülich correlação).
    npsh_a : float
        NPSH disponível na instalação [m].
    sigma_plant : float
        Número de Thoma da instalação: σ = NPSHa / H.
    sigma_critical : float
        Número de Thoma crítico: σ_c = NPSHr / H.
    margin : float
        Margem de cavitação: NPSHa − NPSHr [m].  Positivo = seguro.
    safe : bool
        True se margem ≥ margem_minima.
    nq : float
        Velocidade específica adimensional (si) da bomba.
    suction_specific_speed : float
        Velocidade específica de sucção Nss = n*sqrt(Q) / NPSHr^0.75.
    risk_level : str
        "safe" | "marginal" | "risky" | "critical".
    recommendations : list[str]
        Lista de recomendações de engenharia.
    """

    npsh_r: float
    npsh_a: float
    sigma_plant: float
    sigma_critical: float
    margin: float
    safe: bool
    nq: float
    suction_specific_speed: float
    risk_level: str
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "npsh_r_m": round(self.npsh_r, 3),
            "npsh_a_m": round(self.npsh_a, 3),
            "margin_m": round(self.margin, 3),
            "sigma_plant": round(self.sigma_plant, 4),
            "sigma_critical": round(self.sigma_critical, 4),
            "safe": self.safe,
            "nq": round(self.nq, 2),
            "suction_specific_speed": round(self.suction_specific_speed, 1),
            "risk_level": self.risk_level,
            "recommendations": self.recommendations,
        }


@dataclass
class MinimumPressureResult:
    """Pressão mínima nas superfícies da pá — risco de cavitação local."""

    p_min_ps: float        # Pressão mínima na PS [Pa]
    p_min_ss: float        # Pressão mínima na SS [Pa]
    p_vapor: float         # Pressão de vapor [Pa]
    cavitation_ps: bool    # Cavitação possível na PS (p < p_vap)
    cavitation_ss: bool    # Cavitação possível na SS
    xi_ps_min: float       # Posição do p_min na PS (0–1)
    xi_ss_min: float       # Posição do p_min na SS
    source: str = "estimate"

    def to_dict(self) -> dict:
        return {
            "p_min_ps_pa": round(self.p_min_ps, 0),
            "p_min_ss_pa": round(self.p_min_ss, 0),
            "p_vapor_pa": round(self.p_vapor, 0),
            "cavitation_ps": self.cavitation_ps,
            "cavitation_ss": self.cavitation_ss,
            "xi_ps_min": round(self.xi_ps_min, 3),
            "xi_ss_min": round(self.xi_ss_min, 3),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def assess_cavitation(
    sizing_result,
    npsh_available: float,
    fluid_temp_c: float = 20.0,
    safety_margin: float = 0.5,
) -> CavitationAssessment:
    """Avaliar risco de cavitação dado o NPSHa da instalação.

    Usa correlação de Gülich (2014) eq. 6.10–6.15 para NPSHr.

    Parameters
    ----------
    sizing_result : SizingResult
        Resultado do sizing 1D da bomba.
    npsh_available : float
        NPSH disponível na instalação [m].
    fluid_temp_c : float
        Temperatura do fluido [°C] (afeta pressão de vapor).
    safety_margin : float
        Margem mínima de segurança NPSHa − NPSHr [m].

    Returns
    -------
    CavitationAssessment
    """
    op = sizing_result.op
    Q = op.flow_rate
    H = op.head
    n = op.rpm

    # Velocidade específica (europeia SI: n*sqrt(Q)/H^0.75)
    nq = n * math.sqrt(Q) / H ** 0.75

    # NPSHr — correlação Gülich (2014) eq. 6.10
    # NPSHr = fn * nq^(4/3) * Q^(2/3)  [aproximação dimensional]
    # Versão simplificada: NPSHr = (0.3 + 6.5e-5 * nq^2) * H^0.5
    fn = 0.3 + 6.5e-5 * nq ** 2
    npsh_r = fn * math.sqrt(H)
    npsh_r = max(npsh_r, 0.3)  # mínimo físico

    # Pressão de vapor em função da temperatura
    p_vap = _vapor_pressure(fluid_temp_c)
    # Converter NPSHa para dimensões consistentes
    sigma_plant = npsh_available / max(H, 0.1)
    sigma_crit = npsh_r / max(H, 0.1)

    margin = npsh_available - npsh_r

    # Velocidade específica de sucção (Nss)
    nss = n * math.sqrt(Q) / max(npsh_r, 0.1) ** 0.75

    # Nível de risco
    if margin >= safety_margin:
        risk = "safe"
    elif margin >= 0.0:
        risk = "marginal"
    elif margin >= -1.0:
        risk = "risky"
    else:
        risk = "critical"

    recs = _build_recommendations(nq, nss, margin, sigma_plant, sigma_crit)

    return CavitationAssessment(
        npsh_r=npsh_r,
        npsh_a=npsh_available,
        sigma_plant=sigma_plant,
        sigma_critical=sigma_crit,
        margin=margin,
        safe=margin >= safety_margin,
        nq=nq,
        suction_specific_speed=nss,
        risk_level=risk,
        recommendations=recs,
    )


def assess_minimum_pressure(
    case_dir: "str | Path",
    op: OperatingPoint,
    fluid_temp_c: float = 20.0,
) -> MinimumPressureResult:
    """Extrair pressão mínima nas superfícies da pá dos resultados CFD.

    Tenta ler ``postProcessing/bladePressure/``.  Se não encontrar,
    retorna estimativa analítica baseada na teoria de perfil delgado.

    Parameters
    ----------
    case_dir : Path
        Raiz do caso OpenFOAM.
    op : OperatingPoint
        Ponto de operação.
    fluid_temp_c : float
        Temperatura do fluido [°C] para pressão de vapor.
    """
    case_dir = Path(case_dir)
    p_vap = _vapor_pressure(fluid_temp_c)

    result = _extract_from_postprocessing(case_dir, op, p_vap)
    if result is not None:
        return result

    return _analytical_pressure_estimate(op, p_vap)


# ---------------------------------------------------------------------------
# Extração OpenFOAM
# ---------------------------------------------------------------------------

def _extract_from_postprocessing(
    case_dir: Path,
    op: OperatingPoint,
    p_vap: float,
) -> Optional[MinimumPressureResult]:
    """Tentar extrair p_min do diretório postProcessing/bladePressure/."""
    pp_dir = case_dir / "postProcessing" / "bladePressure"
    if not pp_dir.exists():
        return None

    time_dirs = sorted(
        [d for d in pp_dir.iterdir() if d.is_dir()],
        key=lambda d: float(d.name) if d.name.replace(".", "").isdigit() else 0,
    )
    if not time_dirs:
        return None

    pressure_file = time_dirs[-1] / "surfaceFieldValue.dat"
    if not pressure_file.exists():
        return None

    try:
        xi_list, p_ps_list, p_ss_list = [], [], []
        for line in pressure_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                xi_list.append(float(parts[0]))
                p_ps_list.append(float(parts[1]))
                p_ss_list.append(float(parts[2]))

        if not xi_list:
            return None

        p_min_ps = min(p_ps_list)
        p_min_ss = min(p_ss_list)
        xi_ps_min = xi_list[p_ps_list.index(p_min_ps)]
        xi_ss_min = xi_list[p_ss_list.index(p_min_ss)]

        return MinimumPressureResult(
            p_min_ps=p_min_ps,
            p_min_ss=p_min_ss,
            p_vapor=p_vap,
            cavitation_ps=p_min_ps < p_vap,
            cavitation_ss=p_min_ss < p_vap,
            xi_ps_min=xi_ps_min,
            xi_ss_min=xi_ss_min,
            source="openfoam",
        )
    except (ValueError, OSError):
        return None


def _analytical_pressure_estimate(op: OperatingPoint, p_vap: float) -> MinimumPressureResult:
    """Estimativa analítica de pressão mínima (Bernoulli + aceleração centrífuga)."""
    n = op.rpm
    Q = op.flow_rate
    d1 = getattr(op, "d1", 0.15)
    d2 = getattr(op, "d2", 0.3)

    u1 = math.pi * n / 60.0 * d1
    u2 = math.pi * n / 60.0 * d2

    # Pressão dinâmica de referência na entrada
    A1 = math.pi / 4.0 * d1 ** 2
    c1 = Q / max(A1, 1e-6)
    p_ref = 101325.0  # Pa (atmospheric)

    # Pressão mínima na SS ocorre perto do LE (ponto de sucção máxima)
    # Estimativa: p_min_ss ≈ p_ref − ρ/2 * (u1² + c1²) * factor
    p_min_ss = p_ref - 0.5 * _RHO_WATER * (u1 ** 2 + c1 ** 2) * 1.3
    # PS tem pressão maior que a média
    p_min_ps = p_ref - 0.5 * _RHO_WATER * c1 ** 2 * 0.5

    return MinimumPressureResult(
        p_min_ps=p_min_ps,
        p_min_ss=p_min_ss,
        p_vapor=p_vap,
        cavitation_ps=p_min_ps < p_vap,
        cavitation_ss=p_min_ss < p_vap,
        xi_ps_min=0.1,   # perto do LE na PS
        xi_ss_min=0.15,  # perto do LE na SS
        source="estimate",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vapor_pressure(temp_c: float) -> float:
    """Pressão de vapor da água via Antoine (Pa)."""
    # Antoine: log10(P_mmHg) = A - B/(C+T)  para T em °C
    A, B, C = 8.07131, 1730.63, 233.426
    p_mmhg = 10 ** (A - B / (C + temp_c))
    return p_mmhg * 133.322  # mmHg → Pa


def _build_recommendations(
    nq: float,
    nss: float,
    margin: float,
    sigma: float,
    sigma_crit: float,
) -> list[str]:
    recs = []
    if margin < 0:
        recs.append(
            f"NPSHa insuficiente (margem={margin:.2f} m). "
            "Aumentar nível de sucção ou reduzir perdas na tubulação de sucção."
        )
    if nss > 220:
        recs.append(
            f"Nss={nss:.0f} > 220 — risco elevado de cavitação. "
            "Considerar redução de rotação ou bomba dupla aspiração."
        )
    if nq > 60:
        recs.append(
            "Nq elevado: bomba de fluxo misto. "
            "Verificar cavitação na zona de entrada do rotor."
        )
    if sigma < sigma_crit * 1.2:
        recs.append(
            f"σ_planta ({sigma:.3f}) < 1.2×σ_crítico ({sigma_crit:.3f}). "
            "Margem de segurança insuficiente para operação off-design."
        )
    if not recs:
        recs.append("Condições de cavitação satisfatórias para o ponto de operação nominal.")
    return recs
