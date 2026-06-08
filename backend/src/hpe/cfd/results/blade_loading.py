"""Coeficiente de pressão e carregamento da pá — Fase 13.

Extrai o coeficiente de pressão Cp ao longo da corda da pá (PS e SS)
dos resultados OpenFOAM, detecta separação de camada limite e computa
o diagrama de carregamento (loading diagram).

Usage
-----
    from hpe.cfd.results.blade_loading import extract_blade_loading, BladeLoadingResult

    result = extract_blade_loading(case_dir, op)
    print(result.cp_ps)          # Cp na superfície de pressão
    print(result.loading_peak)   # Pico de carregamento
    print(result.separation_risk)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hpe.core.models import OperatingPoint

log = logging.getLogger(__name__)

_RHO = 998.2
_G   = 9.80665


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BladeLoadingResult:
    """Resultados do coeficiente de pressão e carregamento da pá.

    Attributes
    ----------
    xi : list[float]
        Coordenada adimensional ao longo da corda (0=LE, 1=TE).
    cp_ps : list[float]
        Coeficiente de pressão na superfície de pressão.
    cp_ss : list[float]
        Coeficiente de pressão na superfície de sucção.
    delta_cp : list[float]
        Carregamento local: Cp_PS − Cp_SS (positivo = carregamento correto).
    loading_peak : float
        Pico de ΔCp ao longo da corda.
    loading_integral : float
        Integral ∫ΔCp dξ (proporcional à circulação).
    separation_risk : bool
        True se gradient dCp/dξ > limiar de separação na SS.
    separation_xi : float | None
        Posição adimensional estimada do ponto de separação.
    u_ref : float
        Velocidade de referência usada para normalização [m/s].
    source : str
        "openfoam" se extraído de resultados CFD, "estimate" se analítico.
    """

    xi: list[float]
    cp_ps: list[float]
    cp_ss: list[float]
    delta_cp: list[float]
    loading_peak: float
    loading_integral: float
    separation_risk: bool
    separation_xi: Optional[float]
    u_ref: float
    source: str = "estimate"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "u_ref": round(self.u_ref, 3),
            "loading_peak": round(self.loading_peak, 4),
            "loading_integral": round(self.loading_integral, 4),
            "separation_risk": self.separation_risk,
            "separation_xi": round(self.separation_xi, 3) if self.separation_xi else None,
            "curve": {
                "xi": [round(x, 4) for x in self.xi],
                "cp_ps": [round(c, 4) for c in self.cp_ps],
                "cp_ss": [round(c, 4) for c in self.cp_ss],
                "delta_cp": [round(d, 4) for d in self.delta_cp],
            },
        }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def extract_blade_loading(
    case_dir: "str | Path",
    op: OperatingPoint,
    n_chord: int = 21,
) -> BladeLoadingResult:
    """Extrair carregamento da pá dos resultados OpenFOAM.

    Tenta ler ``postProcessing/bladePressure/`` gerado pelo functionObject
    ``surfaceFieldValue`` configurado para a patch ``blade``.  Se não
    encontrar, retorna uma estimativa analítica baseada em teoria de
    perfil delgado.

    Parameters
    ----------
    case_dir : Path
        Raiz do caso OpenFOAM.
    op : OperatingPoint
        Ponto de operação para normalização Cp.
    n_chord : int
        Pontos na corda para a estimativa analítica (fallback).
    """
    case_dir = Path(case_dir)
    result = _try_extract_from_postprocessing(case_dir, op)
    if result is not None:
        return result

    log.debug("blade_loading: postProcessing not found, using analytical estimate")
    return _analytical_estimate(op, n_chord)


# ---------------------------------------------------------------------------
# Extração OpenFOAM
# ---------------------------------------------------------------------------

def _try_extract_from_postprocessing(
    case_dir: Path,
    op: OperatingPoint,
) -> Optional[BladeLoadingResult]:
    """Tentar extrair Cp do diretório postProcessing/bladePressure/."""
    pp_dir = case_dir / "postProcessing" / "bladePressure"
    if not pp_dir.exists():
        return None

    # Encontrar pasta de tempo mais recente
    time_dirs = sorted(
        [d for d in pp_dir.iterdir() if d.is_dir()],
        key=lambda d: float(d.name) if d.name.replace(".", "").isdigit() else 0,
    )
    if not time_dirs:
        return None

    pressure_file = time_dirs[-1] / "surfaceFieldValue.dat"
    if not pressure_file.exists():
        return None

    # Parse colunas: xi  p_ps  p_ss  (coordenada normalizada + pressões)
    xi_list, p_ps_list, p_ss_list = [], [], []
    try:
        for line in pressure_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                xi_list.append(float(parts[0]))
                p_ps_list.append(float(parts[1]))
                p_ss_list.append(float(parts[2]))
    except (ValueError, OSError) as exc:
        log.warning("blade_loading: parse error %s", exc)
        return None

    if len(xi_list) < 3:
        return None

    # Velocidade de referência: tip speed
    u_ref = math.pi * op.rpm / 60.0 * getattr(op, "d2", 0.3)
    q_ref = 0.5 * _RHO * max(u_ref, 1.0) ** 2

    cp_ps = [(p - p_ps_list[0]) / q_ref for p in p_ps_list]
    cp_ss = [(p - p_ss_list[0]) / q_ref for p in p_ss_list]
    delta = [a - b for a, b in zip(cp_ps, cp_ss)]

    peak = max(delta) if delta else 0.0
    integral = _trapz(xi_list, delta)
    sep_xi = _detect_separation(xi_list, cp_ss)

    return BladeLoadingResult(
        xi=xi_list,
        cp_ps=cp_ps,
        cp_ss=cp_ss,
        delta_cp=delta,
        loading_peak=peak,
        loading_integral=integral,
        separation_risk=sep_xi is not None,
        separation_xi=sep_xi,
        u_ref=u_ref,
        source="openfoam",
    )


# ---------------------------------------------------------------------------
# Estimativa analítica (thin-airfoil + parábola)
# ---------------------------------------------------------------------------

def _analytical_estimate(op: OperatingPoint, n_chord: int) -> BladeLoadingResult:
    """Estimar Cp por teoria de perfil delgado em rampa (Gülich §3.7)."""
    # Ângulos de pá típicos
    beta1 = getattr(op, "beta1_deg", 25.0)
    beta2 = getattr(op, "beta2_deg", 22.0)
    d2 = getattr(op, "d2", 0.3)
    n_rpm = op.rpm

    u_ref = math.pi * n_rpm / 60.0 * d2

    xi = [i / (n_chord - 1) for i in range(n_chord)]
    cp_ps, cp_ss, delta = [], [], []

    for x in xi:
        # Camber line: deflexão linear de beta1 → beta2
        beta = math.radians(beta1 + x * (beta2 - beta1))
        # Cp teórico: pressurização na PS (positivo) e sucção na SS (negativo)
        cp_mid = -2.0 * math.sin(beta) * x  # simplificação de perfil delgado
        cp_thickness = 0.5 * math.cos(beta) * (1 - (2 * x - 1) ** 2)
        ps = cp_mid + cp_thickness
        ss = cp_mid - cp_thickness
        cp_ps.append(ps)
        cp_ss.append(ss)
        delta.append(ps - ss)

    peak = max(delta) if delta else 0.0
    integral = _trapz(xi, delta)
    sep_xi = _detect_separation(xi, cp_ss)

    return BladeLoadingResult(
        xi=xi,
        cp_ps=cp_ps,
        cp_ss=cp_ss,
        delta_cp=delta,
        loading_peak=peak,
        loading_integral=integral,
        separation_risk=sep_xi is not None,
        separation_xi=sep_xi,
        u_ref=u_ref,
        source="estimate",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_separation(
    xi: list[float],
    cp_ss: list[float],
    grad_threshold: float = 0.5,
) -> Optional[float]:
    """Detectar ponto de separação: dCp_SS/dξ > threshold (adverse pressure gradient)."""
    for i in range(1, len(xi)):
        dxi = xi[i] - xi[i - 1]
        if dxi <= 0:
            continue
        grad = (cp_ss[i] - cp_ss[i - 1]) / dxi
        if grad > grad_threshold:
            return xi[i]
    return None


def _trapz(x: list[float], y: list[float]) -> float:
    """Integral trapezoidal simples."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return total
