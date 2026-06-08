"""Curva da bomba H-Q — montagem, ajuste polinomial e BEP — Fase 11.

Recebe os resultados de uma varredura multi-ponto (:class:`SweepResult`) e
produz:

  - Curva H-Q ajustada por polinômio de grau 2 (padrão ISO 9906)
  - Curva η-Q ajustada por parábola
  - Curva P-Q (potência no eixo)
  - Ponto de melhor eficiência (BEP) interpolado
  - Margem de cavitação (NPSHr estimado × fração)

Usage
-----
    from hpe.cfd.pump_curve import build_pump_curve, PumpCurve
    from hpe.cfd.sweep import run_cfd_sweep, SweepConfig

    sweep = run_cfd_sweep(sizing, SweepConfig(), "./sweep_01")
    curve = build_pump_curve(sweep)

    print(curve.bep)
    q_range = [0.03, 0.04, 0.05, 0.06]
    print(curve.H(q_range))
    print(curve.eta(q_range))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hpe.cfd.sweep import SweepResult

log = logging.getLogger(__name__)

# Número mínimo de pontos para ajuste confiável
_MIN_POINTS = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BEP:
    """Ponto de melhor eficiência interpolado.

    Attributes
    ----------
    Q : float
        Vazão no BEP [m³/s].
    H : float
        Altura no BEP [m].
    eta : float
        Eficiência máxima (0–1).
    P_shaft : float
        Potência no eixo no BEP [W].
    fraction : float
        Q_bep / Q_nominal (desvio do ponto de projeto).
    """
    Q: float
    H: float
    eta: float
    P_shaft: float
    fraction: float = 1.0


@dataclass
class PumpCurve:
    """Curva completa da bomba com coeficientes polinomiais ajustados.

    Attributes
    ----------
    Q_points : list[float]
        Pontos de vazão usados no ajuste [m³/s].
    H_points : list[float]
        Pontos de altura (CFD ou estimados) [m].
    eta_points : list[float]
        Pontos de eficiência (0–1).
    P_points : list[float]
        Pontos de potência [W].
    coeff_H : np.ndarray
        Coeficientes do polinômio H(Q) = c[0] + c[1]*Q + c[2]*Q².
        Grau 2 (ISO 9906 characteristic curve).
    coeff_eta : np.ndarray
        Coeficientes do polinômio η(Q).
    coeff_P : np.ndarray
        Coeficientes do polinômio P(Q).
    bep : BEP
        Ponto de melhor eficiência.
    Q_min : float
        Vazão mínima modelada [m³/s].
    Q_max : float
        Vazão máxima modelada [m³/s].
    n_rpm : float
        Velocidade de rotação [rpm].
    npsh_r_bep : float
        NPSHr estimado no BEP [m].
    """

    Q_points: list[float]
    H_points: list[float]
    eta_points: list[float]
    P_points: list[float]
    coeff_H: "np.ndarray"
    coeff_eta: "np.ndarray"
    coeff_P: "np.ndarray"
    bep: BEP
    Q_min: float
    Q_max: float
    n_rpm: float
    npsh_r_bep: float = 0.0

    # ── Avaliação da curva ajustada ──────────────────────────────────────────

    def H(self, Q: "float | list[float] | np.ndarray") -> "np.ndarray":
        """Avaliar H(Q) no(s) ponto(s) especificado(s) [m]."""
        q = np.atleast_1d(np.asarray(Q, dtype=float))
        return np.polyval(self.coeff_H[::-1], q)   # coeff em ordem crescente → polyval inverte

    def eta(self, Q: "float | list[float] | np.ndarray") -> "np.ndarray":
        """Avaliar η(Q) no(s) ponto(s) especificado(s) [0–1]."""
        q = np.atleast_1d(np.asarray(Q, dtype=float))
        return np.clip(np.polyval(self.coeff_eta[::-1], q), 0.0, 1.0)

    def P(self, Q: "float | list[float] | np.ndarray") -> "np.ndarray":
        """Avaliar P_shaft(Q) [W]."""
        q = np.atleast_1d(np.asarray(Q, dtype=float))
        return np.maximum(0.0, np.polyval(self.coeff_P[::-1], q))

    def affinity_scale(self, n_new: float) -> "PumpCurve":
        """Escalar curva para nova rotação via leis de afinidade.

        Q ∝ n,  H ∝ n²,  P ∝ n³

        Parameters
        ----------
        n_new : float
            Nova velocidade de rotação [rpm].

        Returns
        -------
        PumpCurve
            Nova curva escalada (sem refazer o ajuste).
        """
        ratio = n_new / self.n_rpm
        Q_new = [q * ratio for q in self.Q_points]
        H_new = [h * ratio ** 2 for h in self.H_points]
        P_new = [p * ratio ** 3 for p in self.P_points]
        # eta não muda com afinidade
        new_curve = build_pump_curve_from_points(
            Q_pts=Q_new,
            H_pts=H_new,
            eta_pts=list(self.eta_points),
            P_pts=P_new,
            n_rpm=n_new,
        )
        return new_curve

    def to_dict(self) -> dict:
        """Serializar para dicionário (API / frontend-friendly)."""
        q_dense = np.linspace(self.Q_min, self.Q_max, 50)
        return {
            "n_rpm": self.n_rpm,
            "Q_min": self.Q_min,
            "Q_max": self.Q_max,
            "npsh_r_bep": round(self.npsh_r_bep, 3),
            "bep": {
                "Q": round(self.bep.Q, 6),
                "H": round(self.bep.H, 3),
                "eta": round(self.bep.eta, 4),
                "P_kW": round(self.bep.P_shaft / 1000, 3),
                "fraction": round(self.bep.fraction, 3),
            },
            "raw_points": {
                "Q": [round(q, 6) for q in self.Q_points],
                "H": [round(h, 3) for h in self.H_points],
                "eta": [round(e, 4) for e in self.eta_points],
                "P_kW": [round(p / 1000, 3) for p in self.P_points],
            },
            "curve_dense": {
                "Q": q_dense.round(6).tolist(),
                "H": self.H(q_dense).round(3).tolist(),
                "eta": self.eta(q_dense).round(4).tolist(),
                "P_kW": (self.P(q_dense) / 1000).round(3).tolist(),
            },
            "coeff_H": self.coeff_H.tolist(),
            "coeff_eta": self.coeff_eta.tolist(),
            "coeff_P": self.coeff_P.tolist(),
        }


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def build_pump_curve(sweep: "SweepResult") -> PumpCurve:  # type: ignore[name-defined]
    """Construir curva da bomba a partir de um SweepResult.

    Parameters
    ----------
    sweep : SweepResult
        Resultado de :func:`hpe.cfd.sweep.run_cfd_sweep`.

    Returns
    -------
    PumpCurve
        Curva ajustada com BEP e coeficientes.

    Raises
    ------
    ValueError
        Se não há pontos válidos suficientes para ajuste (< 3).
    """
    # Coletar pontos válidos
    Q_pts, H_pts, eta_pts, P_pts = [], [], [], []
    for pt in sweep.points:
        Q_pts.append(pt.Q)
        H_pts.append(pt.H)
        eta_pts.append(pt.eta)
        P_pts.append(pt.P_shaft or _estimate_power(pt.Q, pt.H, pt.eta))

    return build_pump_curve_from_points(
        Q_pts=Q_pts,
        H_pts=H_pts,
        eta_pts=eta_pts,
        P_pts=P_pts,
        n_rpm=sweep.sizing_bep.op.rpm,
        Q_nominal=sweep.Q_bep,
    )


def build_pump_curve_from_points(
    Q_pts: list[float],
    H_pts: list[float],
    eta_pts: list[float],
    P_pts: list[float],
    n_rpm: float,
    Q_nominal: Optional[float] = None,
) -> PumpCurve:
    """Construir curva a partir de listas de pontos (sem SweepResult).

    Útil para montar curvas a partir de dados históricos ou bancada.
    """
    n = len(Q_pts)
    if n < _MIN_POINTS:
        raise ValueError(
            f"Mínimo {_MIN_POINTS} pontos para ajuste da curva; recebidos {n}."
        )

    Q = np.array(Q_pts, dtype=float)
    H = np.array(H_pts, dtype=float)
    eta = np.array(eta_pts, dtype=float)
    P = np.array(P_pts, dtype=float)

    deg = min(2, n - 1)

    # Ajuste polinomial: coeficientes em ordem CRESCENTE (c[0] + c[1]*x + c[2]*x²)
    coeff_H = np.polyfit(Q, H, deg)[::-1]
    coeff_eta = np.polyfit(Q, eta, deg)[::-1]
    coeff_P = np.polyfit(Q, P, deg)[::-1]

    # BEP: máximo de η na faixa de Q
    Q_dense = np.linspace(Q.min(), Q.max(), 500)
    eta_dense = np.clip(np.polyval(coeff_eta[::-1], Q_dense), 0, 1)
    idx_bep = int(np.argmax(eta_dense))
    Q_bep = float(Q_dense[idx_bep])
    H_bep = float(np.polyval(coeff_H[::-1], Q_bep))
    eta_bep = float(eta_dense[idx_bep])
    P_bep = float(max(0.0, np.polyval(coeff_P[::-1], Q_bep)))
    frac_bep = Q_bep / Q_nominal if Q_nominal else 1.0

    bep = BEP(Q=Q_bep, H=H_bep, eta=eta_bep, P_shaft=P_bep, fraction=frac_bep)

    # NPSHr estimado no BEP (Gülich eq. 6.24: NPSHr ≈ 0.3 × H_bep^0.5 + 0.1)
    npsh_r = 0.3 * math.sqrt(H_bep) + 0.1 if H_bep > 0 else 0.0

    log.info(
        "Pump curve: BEP Q=%.4f H=%.1f η=%.1f%%  NPSHr≈%.1f m",
        Q_bep, H_bep, eta_bep * 100, npsh_r,
    )

    return PumpCurve(
        Q_points=list(Q_pts),
        H_points=list(H_pts),
        eta_points=list(eta_pts),
        P_points=list(P_pts),
        coeff_H=coeff_H,
        coeff_eta=coeff_eta,
        coeff_P=coeff_P,
        bep=bep,
        Q_min=float(Q.min()),
        Q_max=float(Q.max()),
        n_rpm=n_rpm,
        npsh_r_bep=npsh_r,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_power(Q: float, H: float, eta: float, rho: float = 998.2) -> float:
    """Estimar P_shaft a partir de Q, H, η."""
    g = 9.80665
    if eta <= 0:
        return 0.0
    return rho * g * Q * H / eta
