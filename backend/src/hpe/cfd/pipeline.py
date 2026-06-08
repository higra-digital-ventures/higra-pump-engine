"""HPE CFD Pipeline — principal entry point da Fase 2.

Orquestra o fluxo completo: sizing → geometria → caso OpenFOAM → solver →
extração de resultados → training_log.

Funciona sem OpenFOAM instalado: gera os arquivos de caso mas não tenta
rodar o solver automaticamente a menos que `run_solver=True` e `openfoam`
esteja no PATH.

Usage
-----
    from hpe.cfd.pipeline import run_cfd_pipeline, CfdResult
    from hpe.core.models import OperatingPoint
    from hpe.sizing.meanline import run_sizing

    op = OperatingPoint(flow_rate=0.05, head=30, rpm=1750)
    sizing = run_sizing(op)
    result = run_cfd_pipeline(sizing, output_dir="./cases/pump_01")
    print(result.openfoam_available, result.ran_simulation)
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hpe.core.models import OperatingPoint, PerformanceMetrics, SizingResult
from hpe.geometry.parametric import run_geometry
from hpe.geometry.models import RunnerGeometryParams

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CfdResult:
    """Resultado completo do pipeline CFD para um ponto de operação.

    Attributes
    ----------
    case_dir : str
        Caminho absoluto para o diretório do caso OpenFOAM gerado.
    step_file : str | None
        Caminho para o arquivo STEP exportado, ou None se CadQuery não disponível.
    openfoam_available : bool
        True se o OpenFOAM foi detectado no PATH do sistema.
    ran_simulation : bool
        True se o solver foi executado com sucesso.
    performance : PerformanceMetrics | None
        Métricas extraídas dos resultados CFD, ou None se não rodou.
    errors : list[str]
        Erros e avisos não-fatais coletados durante o pipeline.
    training_log_id : str | None
        UUID da linha inserida em hpe.training_log, ou None se não inserido.
    """

    case_dir: str
    step_file: Optional[str]
    openfoam_available: bool
    ran_simulation: bool
    performance: Optional[PerformanceMetrics]
    errors: list[str]
    training_log_id: Optional[str]

    def summary(self) -> str:
        """Resumo legível do resultado do pipeline."""
        lines = [
            "=== HPE CFD Pipeline Result ===",
            f"  Case dir  : {self.case_dir}",
            f"  STEP file : {self.step_file or 'N/A'}",
            f"  OpenFOAM  : {'available' if self.openfoam_available else 'NOT available'}",
            f"  Simulated : {'YES' if self.ran_simulation else 'NO'}",
        ]
        if self.performance:
            p = self.performance
            lines += [
                f"  H         : {p.head:.2f} m",
                f"  eta_total : {p.total_efficiency * 100:.1f} %",
                f"  Power     : {p.power / 1000:.2f} kW",
            ]
        if self.training_log_id:
            lines.append(f"  Log ID    : {self.training_log_id}")
        if self.errors:
            lines.append("  Errors/Warnings:")
            for e in self.errors:
                lines.append(f"    ! {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------


def run_cfd_pipeline(
    sizing: SizingResult,
    output_dir: str,
    run_solver: bool = False,
    n_procs: int = 4,
) -> CfdResult:
    """Executa o pipeline CFD completo para um resultado de sizing 1D.

    Pipeline
    --------
    1. Gera RunnerGeometryParams via hpe.geometry.parametric.run_geometry()
    2. Cria estrutura de diretórios do caso OpenFOAM
    3. Escreve todos os arquivos de configuração via hpe.cfd.openfoam.*
    4. Tenta exportar STEP (silencioso se CadQuery não disponível)
    5. Se run_solver=True e openfoam no PATH → executa blockMesh && MRFSimpleFoam
    6. Extrai performance se simulação rodou
    7. Se performance disponível, insere no training_log (fonte='cfd_openfoam')
    8. Retorna CfdResult

    Parameters
    ----------
    sizing : SizingResult
        Resultado do sizing 1D (hpe.sizing.meanline.run_sizing).
    output_dir : str
        Diretório onde criar o caso OpenFOAM.
    run_solver : bool
        Se True e OpenFOAM estiver no PATH, executa a simulação.
    n_procs : int
        Número de processos para execução paralela.

    Returns
    -------
    CfdResult
        Resultado completo com case_dir, step_file, métricas e log ID.
    """
    errors: list[str] = []
    case_dir = Path(output_dir).resolve()
    step_file: Optional[str] = None
    performance: Optional[PerformanceMetrics] = None
    training_log_id: Optional[str] = None
    ran_simulation = False

    # ------------------------------------------------------------------
    # 1. Geometria paramétrica
    # ------------------------------------------------------------------
    log.info("CFD Pipeline: Step 1 — generating geometry params")
    try:
        geo_result = run_geometry(
            sizing,
            export_dir=case_dir / "geometry",
            export_step=True,
            export_stl=False,
        )
        params = geo_result.params
        if geo_result.step_path:
            step_file = geo_result.step_path
            log.info("CFD Pipeline: STEP exported to %s", step_file)
        if geo_result.warnings:
            errors.extend([f"[geometry] {w}" for w in geo_result.warnings])
    except Exception as exc:
        errors.append(f"[geometry] failed: {exc}")
        log.exception("CFD Pipeline: geometry generation failed")
        # Fallback: build params directly from sizing
        params = RunnerGeometryParams.from_sizing_result(sizing)

    # ------------------------------------------------------------------
    # 2. Recuperar OperatingPoint a partir do SizingResult
    # ------------------------------------------------------------------
    op = _op_from_sizing(sizing)

    # ------------------------------------------------------------------
    # 3. Criar estrutura do caso OpenFOAM + arquivos de configuração
    # ------------------------------------------------------------------
    log.info("CFD Pipeline: Step 2 — building OpenFOAM case in %s", case_dir)
    try:
        from hpe.cfd.openfoam.case import build_openfoam_case

        created_files = build_openfoam_case(
            params=params,
            op=op,
            case_dir=case_dir,
            n_procs=n_procs,
        )
        log.info("CFD Pipeline: %d files created in case dir", len(created_files))
    except Exception as exc:
        errors.append(f"[openfoam_case] failed: {exc}")
        log.exception("CFD Pipeline: OpenFOAM case build failed")

    # ------------------------------------------------------------------
    # 4. Copiar STEP/STL para constant/triSurface/
    # ------------------------------------------------------------------
    if step_file and Path(step_file).exists():
        tri_dir = case_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(step_file, tri_dir / Path(step_file).name)
        except Exception as exc:
            errors.append(f"[step_copy] {exc}")

    # ------------------------------------------------------------------
    # 5. Verificar disponibilidade do OpenFOAM
    # ------------------------------------------------------------------
    openfoam_available = _check_openfoam()

    # ------------------------------------------------------------------
    # 6. Executar simulação (opcional)
    # ------------------------------------------------------------------
    if run_solver:
        if not openfoam_available:
            errors.append(
                "run_solver=True mas OpenFOAM não encontrado no PATH. "
                "Execute manualmente: cd {case_dir} && ./run.sh"
            )
        else:
            log.info("CFD Pipeline: Step 3 — running OpenFOAM solver")
            ran_simulation, solver_errors = _run_openfoam(case_dir, n_procs)
            errors.extend(solver_errors)

    # ------------------------------------------------------------------
    # 7. Extrair performance
    # ------------------------------------------------------------------
    if ran_simulation:
        try:
            from hpe.cfd.results.extract import extract_performance

            cfd_perf = extract_performance(case_dir, op)
            if cfd_perf.converged:
                performance = _cfd_perf_to_metrics(cfd_perf, sizing)
                log.info(
                    "CFD Pipeline: performance extracted — H=%.1fm, eta=%.1f%%",
                    performance.head,
                    performance.total_efficiency * 100,
                )
            else:
                errors.append("[extract] simulation did not converge")
        except Exception as exc:
            errors.append(f"[extract] failed: {exc}")
            log.exception("CFD Pipeline: result extraction failed")

    # ------------------------------------------------------------------
    # 8. Registrar no training_log
    # ------------------------------------------------------------------
    if performance is not None:
        try:
            training_log_id = _insert_training_log(sizing, op, performance)
            log.info("CFD Pipeline: training_log ID = %s", training_log_id)
        except Exception as exc:
            errors.append(f"[training_log] failed: {exc}")
            log.warning("CFD Pipeline: training_log insert failed — %s", exc)

    return CfdResult(
        case_dir=str(case_dir),
        step_file=step_file,
        openfoam_available=openfoam_available,
        ran_simulation=ran_simulation,
        performance=performance,
        errors=errors,
        training_log_id=training_log_id,
    )


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _op_from_sizing(sizing: SizingResult) -> OperatingPoint:
    """Reconstruir OperatingPoint aproximado a partir do SizingResult."""
    # Extrair rpm a partir do triângulo de velocidades
    try:
        u2 = sizing.velocity_triangles.get("outlet", {}).get("u", None)
        if u2 and sizing.impeller_d2 > 0:
            rpm = 60.0 * u2 / (math.pi * sizing.impeller_d2)
        else:
            rpm = 1450.0  # fallback razoável
    except Exception:
        rpm = 1450.0

    # Extrair Q e H do meridional_profile se disponível
    try:
        mp = sizing.meridional_profile
        q_approx = mp.get("q", None)
        h_approx = mp.get("h", None)
    except Exception:
        q_approx = None
        h_approx = None

    # Estimar Q a partir da velocidade meridional e área de entrada
    if q_approx is None:
        d1 = sizing.impeller_d1
        d1h = sizing.meridional_profile.get("d1_hub", d1 * 0.35) if hasattr(sizing, "meridional_profile") else d1 * 0.35
        a1 = math.pi / 4.0 * (d1**2 - d1h**2)
        # cm1 típico: ~3-5 m/s, usar cm da triangulo de velocidades se disponível
        try:
            cm1 = sizing.velocity_triangles.get("inlet", {}).get("cm", 4.0)
        except Exception:
            cm1 = 4.0
        q_approx = a1 * cm1

    # Estimar H a partir da potência e eficiência estimada
    if h_approx is None:
        try:
            # H = P_shaft * eta / (rho * g * Q)
            rho = 998.2
            g = 9.80665
            eta = max(sizing.estimated_efficiency, 0.3)
            h_approx = sizing.estimated_power * eta / (rho * g * max(q_approx, 1e-6))
        except Exception:
            h_approx = 30.0  # fallback

    return OperatingPoint(
        flow_rate=q_approx,
        head=h_approx,
        rpm=rpm,
    )


def _check_openfoam() -> bool:
    """Verificar se OpenFOAM está disponível no PATH."""
    try:
        import subprocess
        result = subprocess.run(
            ["blockMesh", "-help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return True
    except (FileNotFoundError, OSError):
        return False
    except Exception:
        return False


def _run_openfoam(case_dir: Path, n_procs: int) -> tuple[bool, list[str]]:
    """Executar blockMesh + MRFSimpleFoam no caso especificado.

    Returns
    -------
    tuple[bool, list[str]]
        (sucesso, lista_de_erros)
    """
    import subprocess

    errors: list[str] = []
    ran = False

    def _run(cmd: list[str]) -> tuple[bool, str]:
        try:
            res = subprocess.run(
                cmd,
                cwd=str(case_dir),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            return res.returncode == 0, res.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command {cmd[0]} timed out"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"

    # blockMesh
    ok, err = _run(["blockMesh"])
    if not ok:
        errors.append(f"[blockMesh] {err[:200]}")
        return False, errors

    # snappyHexMesh (opcional — pode falhar se STL não disponível)
    ok, err = _run(["snappyHexMesh", "-overwrite"])
    if not ok:
        errors.append(f"[snappyHexMesh] {err[:200]} (non-fatal)")

    # decomposePar se paralelo
    if n_procs > 1:
        ok, err = _run(["decomposePar", "-force"])
        if not ok:
            errors.append(f"[decomposePar] {err[:200]}")
            return False, errors

        ok, err = _run(["mpirun", "-np", str(n_procs), "MRFSimpleFoam", "-parallel"])
        if ok:
            _run(["reconstructPar", "-latestTime"])
    else:
        ok, err = _run(["MRFSimpleFoam"])

    if ok:
        ran = True
    else:
        errors.append(f"[MRFSimpleFoam] {err[:300]}")

    return ran, errors


def _cfd_perf_to_metrics(cfd_perf, sizing: SizingResult) -> PerformanceMetrics:
    """Converter CfdPerformance → PerformanceMetrics."""
    rho = 998.2
    g = 9.80665
    q = cfd_perf.Q
    h = cfd_perf.H
    p_shaft = cfd_perf.P_shaft

    hydraulic_power = rho * g * q * h
    eta_total = hydraulic_power / p_shaft if p_shaft > 0 else 0.0

    omega = cfd_perf.n_rpm * math.pi / 30.0
    torque = p_shaft / omega if omega > 0 else 0.0

    return PerformanceMetrics(
        hydraulic_efficiency=cfd_perf.eta_total,
        volumetric_efficiency=0.97,    # estimativa padrão
        mechanical_efficiency=0.99,    # estimativa padrão
        total_efficiency=eta_total,
        head=h,
        torque=torque,
        power=p_shaft,
        npsh_required=sizing.estimated_npsh_r,
        min_pressure_coefficient=0.0,
        is_unstable=False,
    )


def _insert_training_log(
    sizing: SizingResult,
    op: OperatingPoint,
    performance: PerformanceMetrics,
) -> Optional[str]:
    """Inserir resultado CFD no training_log.

    Returns UUID da linha inserida, ou None em caso de falha de conexão.
    """
    try:
        from hpe.data.training_log import TrainingLogEntry, insert_entry

        ns = sizing.specific_speed_ns
        entry = TrainingLogEntry(
            fonte="cfd_openfoam",
            ns=ns,
            nq=sizing.specific_speed_nq,
            d2_mm=sizing.impeller_d2 * 1000,
            d1_mm=sizing.impeller_d1 * 1000,
            b2_mm=sizing.impeller_b2 * 1000,
            beta1_deg=sizing.beta1,
            beta2_deg=sizing.beta2,
            z_palhetas=sizing.blade_count,
            n_rpm=op.rpm,
            q_m3h=op.flow_rate * 3600,
            h_m=performance.head,
            eta_total=performance.total_efficiency * 100,
            eta_hid=performance.hydraulic_efficiency * 100,
            p_shaft_kw=performance.power / 1000,
            npsh_r_m=performance.npsh_required,
            qualidade=0.95,
            notas="auto-logged from cfd_openfoam pipeline",
        )
        return insert_entry(entry)
    except ImportError:
        log.warning("training_log not available — skipping insert")
        return None
