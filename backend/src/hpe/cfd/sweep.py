"""Multi-point CFD operating sweep — Fase 11.

Executa o pipeline CFD em múltiplos pontos operacionais (50%–130% Q_bep)
para construção da curva da bomba H-Q e curva de eficiência η-Q.

Usage
-----
    from hpe.cfd.sweep import run_cfd_sweep, SweepConfig
    from hpe.sizing.meanline import run_sizing
    from hpe.core.models import OperatingPoint

    op_bep = OperatingPoint(flow_rate=0.05, head=30, rpm=1750)
    sizing = run_sizing(op_bep)

    config = SweepConfig(flow_fractions=[0.6, 0.8, 1.0, 1.2])
    sweep = run_cfd_sweep(sizing, config, output_dir="./sweep_01")
    print(sweep.to_dataframe())
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hpe.core.models import OperatingPoint, SizingResult

log = logging.getLogger(__name__)

# Frações padrão de Q_bep para varredura da curva completa
DEFAULT_FLOW_FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Parâmetros de varredura multi-ponto.

    Attributes
    ----------
    flow_fractions : list[float]
        Frações de Q_bep a simular.  Default: 0.5 a 1.3 em 9 pontos.
    run_solver : bool
        Se True, executa o solver OpenFOAM em cada ponto.
        Se False, gera apenas os arquivos de caso.
    n_procs : int
        Processadores por simulação (MPI).
    max_workers : int
        Simulações paralelas (via ThreadPoolExecutor).
        Use 1 para rodar em série (recomendado se n_procs > 1).
    mesh_mode : str
        "snappy" ou "structured_blade".
    turbulence_model : str
        "kEpsilon" ou "kOmegaSST".
    n_iter : int
        Máximo de iterações por ponto.
    convergence_tol : float
        Tolerância de resíduo para terminação antecipada.
    """

    flow_fractions: list[float] = field(
        default_factory=lambda: list(DEFAULT_FLOW_FRACTIONS)
    )
    run_solver: bool = False
    n_procs: int = 4
    max_workers: int = 1
    mesh_mode: str = "snappy"
    turbulence_model: str = "kEpsilon"
    n_iter: int = 500
    convergence_tol: float = 1e-4


# ---------------------------------------------------------------------------
# Resultado por ponto
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """Resultado CFD de um único ponto operacional.

    Attributes
    ----------
    fraction : float
        Fração de Q_bep deste ponto (ex.: 0.8 = 80% BEP).
    Q : float
        Vazão volumétrica [m³/s].
    H_target : float
        Altura nominal (da curva de sizing 1D) [m].
    H_cfd : float | None
        Altura extraída do CFD [m], ou None se não simulado.
    eta_cfd : float | None
        Eficiência total extraída do CFD (0–1), ou None.
    P_shaft : float | None
        Potência no eixo [W], ou None.
    converged : bool
        True se a simulação convergiu.
    case_dir : str
        Caminho do diretório do caso OpenFOAM.
    training_log_id : str | None
        ID da linha inserida em hpe.training_log.
    error : str | None
        Mensagem de erro se o ponto falhou.
    """

    fraction: float
    Q: float
    H_target: float
    H_cfd: Optional[float] = None
    eta_cfd: Optional[float] = None
    P_shaft: Optional[float] = None
    converged: bool = False
    case_dir: str = ""
    training_log_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def H(self) -> float:
        """Melhor estimativa de H: CFD se disponível, caso contrário 1D."""
        return self.H_cfd if self.H_cfd is not None else self.H_target

    @property
    def eta(self) -> float:
        """Eficiência: CFD se disponível, caso contrário estimativa 1D."""
        return self.eta_cfd if self.eta_cfd is not None else _estimate_eta_1d(self.fraction)


# ---------------------------------------------------------------------------
# Resultado completo da varredura
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Resultado completo de uma varredura multi-ponto.

    Attributes
    ----------
    bep_fraction : float
        Fração de Q_bep identificada como BEP real (η máximo).
    points : list[SweepPoint]
        Resultados de cada ponto operacional, ordenados por Q.
    sizing_bep : SizingResult
        Resultado de sizing do ponto de projeto (BEP nominal).
    n_converged : int
        Número de pontos que convergiram.
    errors : list[str]
        Erros não-fatais coletados.
    """

    points: list[SweepPoint]
    sizing_bep: SizingResult
    n_converged: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def bep_fraction(self) -> float:
        """Fração de Q onde η é máximo."""
        valid = [p for p in self.points if p.eta_cfd is not None]
        if not valid:
            return 1.0
        return max(valid, key=lambda p: p.eta_cfd).fraction

    @property
    def Q_bep(self) -> float:
        """Vazão BEP nominal [m³/s]."""
        return self.sizing_bep.op.flow_rate

    @property
    def H_bep(self) -> float:
        """Altura BEP nominal [m]."""
        return self.sizing_bep.op.head

    def to_dict(self) -> dict:
        """Serializar para dicionário (API-friendly)."""
        return {
            "bep_fraction": self.bep_fraction,
            "Q_bep": self.Q_bep,
            "H_bep": self.H_bep,
            "n_converged": self.n_converged,
            "n_points": len(self.points),
            "errors": self.errors,
            "points": [
                {
                    "fraction": p.fraction,
                    "Q": round(p.Q, 6),
                    "H_target": round(p.H_target, 3),
                    "H_cfd": round(p.H_cfd, 3) if p.H_cfd is not None else None,
                    "eta_cfd": round(p.eta_cfd, 4) if p.eta_cfd is not None else None,
                    "P_shaft_kW": round(p.P_shaft / 1000, 3) if p.P_shaft else None,
                    "converged": p.converged,
                    "case_dir": p.case_dir,
                    "training_log_id": p.training_log_id,
                    "error": p.error,
                }
                for p in self.points
            ],
        }


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def run_cfd_sweep(
    sizing_bep: SizingResult,
    config: SweepConfig,
    output_dir: str | Path,
) -> SweepResult:
    """Executar varredura CFD multi-ponto ao longo da curva H-Q.

    Para cada fração em ``config.flow_fractions``, cria um OperatingPoint
    modificado (Q = fração × Q_bep, H estimado pela lei de afinidade angular
    + correção de slip), constrói o caso OpenFOAM e opcionalmente executa
    o solver.

    Parameters
    ----------
    sizing_bep : SizingResult
        Resultado de sizing do ponto de projeto (BEP).
    config : SweepConfig
        Parâmetros de varredura (frações, solver, paralelismo).
    output_dir : str | Path
        Diretório raiz onde criar subdiretórios para cada ponto.

    Returns
    -------
    SweepResult
        Resultados de todos os pontos operacionais.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Q_bep = sizing_bep.op.flow_rate
    H_bep = sizing_bep.op.head
    n_rpm = sizing_bep.op.rpm

    log.info(
        "CFD sweep: Q_bep=%.4f m³/s  H_bep=%.1f m  n=%d rpm  %d points",
        Q_bep, H_bep, n_rpm, len(config.flow_fractions),
    )

    # Preparar lista de pontos a simular
    sweep_points: list[SweepPoint] = []
    for frac in sorted(config.flow_fractions):
        Q = Q_bep * frac
        H_est = _estimate_head_at_fraction(H_bep, frac)
        sweep_points.append(SweepPoint(fraction=frac, Q=Q, H_target=H_est))

    # Executar pontos (em série ou paralelo)
    if config.max_workers > 1:
        _run_parallel(sweep_points, sizing_bep, config, output_dir)
    else:
        for pt in sweep_points:
            _run_single_point(pt, sizing_bep, config, output_dir)

    n_conv = sum(1 for p in sweep_points if p.converged)
    errors = [p.error for p in sweep_points if p.error]

    log.info(
        "CFD sweep done: %d/%d converged",
        n_conv, len(sweep_points),
    )

    return SweepResult(
        points=sweep_points,
        sizing_bep=sizing_bep,
        n_converged=n_conv,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Execução por ponto
# ---------------------------------------------------------------------------

def _run_single_point(
    pt: SweepPoint,
    sizing_bep: SizingResult,
    config: SweepConfig,
    output_dir: Path,
) -> None:
    """Rodar pipeline CFD para um único ponto operacional."""
    label = f"pt_{pt.fraction:.2f}".replace(".", "p")
    case_dir = output_dir / label
    pt.case_dir = str(case_dir)

    try:
        from hpe.cfd.pipeline import run_cfd_pipeline

        # Criar OperatingPoint modificado
        op = OperatingPoint(
            flow_rate=pt.Q,
            head=pt.H_target,
            rpm=sizing_bep.op.rpm,
            fluid_density=getattr(sizing_bep.op, "fluid_density", 998.2),
            fluid_viscosity=getattr(sizing_bep.op, "fluid_viscosity", 1.002e-3),
        )

        # Re-sizing para Q diferente do BEP (geometria fixa, ponto off-design)
        # Usamos o sizing do BEP diretamente e apenas ajustamos Q no OperatingPoint
        result = run_cfd_pipeline(
            sizing=sizing_bep,
            output_dir=case_dir,
            run_solver=config.run_solver,
            n_procs=config.n_procs,
            mesh_mode=config.mesh_mode,
            turbulence_model=config.turbulence_model,
            n_iter=config.n_iter,
            operating_point_override=op,
        )

        if result.performance:
            pt.H_cfd = result.performance.head
            pt.eta_cfd = result.performance.eta_total
            pt.P_shaft = result.performance.shaft_power
            pt.converged = getattr(result.performance, "converged", result.ran_simulation)
        else:
            pt.converged = False

        pt.training_log_id = result.training_log_id

    except Exception as exc:
        pt.error = str(exc)
        log.error("Sweep point Q=%.4f failed: %s", pt.Q, exc)


def _run_parallel(
    points: list[SweepPoint],
    sizing_bep: SizingResult,
    config: SweepConfig,
    output_dir: Path,
) -> None:
    """Rodar pontos em paralelo com ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(_run_single_point, pt, sizing_bep, config, output_dir): pt
            for pt in points
        }
        for future in as_completed(futures):
            pt = futures[future]
            try:
                future.result()
            except Exception as exc:
                pt.error = str(exc)
                log.error("Parallel sweep point Q=%.4f error: %s", pt.Q, exc)


# ---------------------------------------------------------------------------
# Estimativas analíticas (fallback quando solver não executa)
# ---------------------------------------------------------------------------

def _estimate_head_at_fraction(H_bep: float, fraction: float) -> float:
    """Estimar H fora do BEP usando curva parabólica típica de bomba centrífuga.

    Approximation: H(Q) = H_bep * (a + b*f + c*f²)
    onde a=1.25, b=-0.05, c=-0.20 (coeficientes típicos Gülich Tab. 4.1).
    Válido para 0.4 ≤ f ≤ 1.3.
    """
    f = fraction
    a, b, c = 1.25, -0.05, -0.20
    return H_bep * max(0.1, a + b * f + c * f * f)


def _estimate_eta_1d(fraction: float) -> float:
    """Estimar η fora do BEP via parábola η-Q típica.

    η(f) = η_bep * (2f - f²) com η_bep = 0.78
    (parábola simétrica em f=1, zero em f=0).
    """
    f = max(0.0, min(1.5, fraction))
    return 0.78 * max(0.0, 2 * f - f * f)
