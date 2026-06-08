"""Regime transiente — sliding mesh + pimpleFoam — Fase 19.1.

Substitui a abordagem MRF (Multiple Reference Frame — steady state)
por sliding mesh rotativa, habilitando análise transiente para:
  - Pulsações de pressão na interface rotor-voluta (Blade Passing Frequency)
  - Forças radiais não-balanceadas variáveis no tempo
  - Ruído hidrodinâmico (BPF, 2×BPF, 3×BPF)
  - Vortices instáveis / stall

Usage
-----
    from hpe.cfd.openfoam.transient import build_transient_case, TransientConfig

    cfg = TransientConfig(end_time=0.1, write_interval=0.002, max_co=2.0)
    case = build_transient_case(sizing, output_dir, cfg)
    # Run: pimpleFoam -case <output_dir>
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TransientConfig:
    """Configuração de caso transiente pimpleFoam.

    Attributes
    ----------
    end_time : float
        Tempo final de simulação [s].  Para FFT em BPF, usar ≥ 10 revoluções.
    write_interval : float
        Intervalo de gravação [s].  Para FFT, precisa capturar ~10 pts/período BPF.
    max_co : float
        Courant number máximo (PIMPLE allows Co > 1, tipicamente 2-5).
    delta_t : float
        Passo inicial [s].  Se 0, calculado automaticamente.
    n_outer_correctors : int
        PIMPLE outer iterations (1 = PISO, >1 = PIMPLE properly).
    n_inner_correctors : int
        PISO-like inner correctors.
    write_format : str
        "binary" | "ascii".
    ddt_scheme : str
        Esquema temporal: "Euler" (1ª ordem) | "backward" (2ª ordem).
    """
    end_time: float = 0.2
    write_interval: float = 0.002
    max_co: float = 2.0
    delta_t: float = 0.0
    n_outer_correctors: int = 2
    n_inner_correctors: int = 2
    write_format: str = "binary"
    ddt_scheme: str = "backward"


@dataclass
class TransientCase:
    """Caso transiente montado."""
    case_dir: Path
    config: TransientConfig
    rpm: float
    n_revolutions: float
    bpf_hz: float           # Blade Passing Frequency
    expected_fs_hz: float   # Sampling frequency (1/write_interval)
    created: bool = False

    def to_dict(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "rpm": self.rpm,
            "end_time": self.config.end_time,
            "write_interval": self.config.write_interval,
            "n_revolutions": round(self.n_revolutions, 2),
            "bpf_hz": round(self.bpf_hz, 2),
            "expected_fs_hz": round(self.expected_fs_hz, 2),
            "max_co": self.config.max_co,
            "ddt_scheme": self.config.ddt_scheme,
            "created": self.created,
        }


def build_transient_case(
    sizing,
    output_dir: "str | Path",
    config: TransientConfig,
    n_procs: int = 4,
) -> TransientCase:
    """Montar caso OpenFOAM transiente com sliding mesh.

    Gera:
      - constant/dynamicMeshDict (cell zone rotativa)
      - system/controlDict com application=pimpleFoam
      - system/fvSchemes com ddt ajustado
      - system/fvSolution com PIMPLE block
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construir caso base via case.py (MRF → será sobrescrito)
    from hpe.cfd.openfoam.case import build_openfoam_case
    build_openfoam_case(
        sizing=sizing,
        output_dir=output_dir,
        mesh_mode="snappy",
        turbulence_model="kOmegaSST",
        n_procs=n_procs,
    )

    rpm = float(getattr(sizing, "n", 1750))
    omega = 2 * math.pi * rpm / 60.0
    blade_count = int(getattr(sizing, "blade_count", 6))
    bpf = blade_count * rpm / 60.0
    fs = 1.0 / config.write_interval if config.write_interval > 0 else 0.0
    n_rev = config.end_time * rpm / 60.0

    # Verificar Nyquist: fs > 2·BPF
    if fs < 2 * bpf:
        log.warning(
            "write_interval=%g não satisfaz Nyquist para BPF=%.1f Hz (fs=%.1f)",
            config.write_interval, bpf, fs,
        )

    # ── Escrever arquivos transientes ───────────────────────────────────────
    _write_dynamic_mesh_dict(output_dir, omega=omega)
    _write_control_dict_transient(output_dir, config)
    _write_fv_schemes_transient(output_dir, config)
    _write_fv_solution_pimple(output_dir, config)

    case = TransientCase(
        case_dir=output_dir,
        config=config,
        rpm=rpm,
        n_revolutions=n_rev,
        bpf_hz=bpf,
        expected_fs_hz=fs,
        created=True,
    )
    log.info(
        "Transient case: end=%.2fs (%.1f rev), BPF=%.1f Hz, fs=%.1f Hz",
        config.end_time, n_rev, bpf, fs,
    )
    return case


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_dynamic_mesh_dict(case_dir: Path, omega: float) -> None:
    """dynamicMeshDict com cellZone rotativa."""
    (case_dir / "constant" / "dynamicMeshDict").write_text(
        f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object dynamicMeshDict; }}

dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolverLibs ("libfvMotionSolvers.so");

motionSolver    solidBody;

solidBodyMotionFunction rotatingMotion;

rotatingMotionCoeffs
{{
    origin      (0 0 0);
    axis        (0 0 1);
    omega       {omega:.4f};       // [rad/s] = 2π×rpm/60
}}

cellZone        rotatingZone;
""",
        encoding="utf-8",
    )


def _write_control_dict_transient(case_dir: Path, cfg: TransientConfig) -> None:
    delta_t = cfg.delta_t if cfg.delta_t > 0 else cfg.write_interval / 20
    (case_dir / "system" / "controlDict").write_text(
        f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}

application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.end_time};

deltaT          {delta_t:.6g};

writeControl    adjustableRunTime;
writeInterval   {cfg.write_interval};
purgeWrite      20;
writeFormat     {cfg.write_format};
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;

runTimeModifiable   true;
adjustTimeStep      yes;
maxCo               {cfg.max_co};
maxDeltaT           {cfg.write_interval};

functions
{{
    probes
    {{
        type            probes;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (p U);
        probeLocations
        (
            (0.15 0.0  0.0)
            (0.0  0.15 0.0)
            (-0.15 0.0 0.0)
            (0.0  -0.15 0.0)
            (0.20  0.0 0.0)  // volute tongue proximity
        );
    }}

    forces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        patches         (blade hub shroud);
        rho             rhoInf;
        rhoInf          998.2;
        CofR            (0 0 0);
    }}
}}
""",
        encoding="utf-8",
    )


def _write_fv_schemes_transient(case_dir: Path, cfg: TransientConfig) -> None:
    (case_dir / "system" / "fvSchemes").write_text(
        f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object fvSchemes; }}

ddtSchemes
{{
    default         {cfg.ddt_scheme};
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}

divSchemes
{{
    default         none;
    div(phi,U)      Gauss linearUpwindV grad(U);
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(phi,nuTilda) Gauss upwind;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}
""",
        encoding="utf-8",
    )


def _write_fv_solution_pimple(case_dir: Path, cfg: TransientConfig) -> None:
    (case_dir / "system" / "fvSolution").write_text(
        f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object fvSolution; }}

solvers
{{
    "pcorr.*"
    {{
        solver          GAMG;
        tolerance       1e-5;
        relTol          0;
        smoother        DICGaussSeidel;
    }}

    p
    {{
        $pcorr;
        tolerance       1e-6;
        relTol          0.01;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    "(U|k|omega)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }}

    "(U|k|omega)Final"
    {{
        $U;
        relTol          0;
    }}
}}

PIMPLE
{{
    correctPhi          yes;
    nOuterCorrectors    {cfg.n_outer_correctors};
    nCorrectors         {cfg.n_inner_correctors};
    nNonOrthogonalCorrectors 1;
    pRefCell            0;
    pRefValue           0;
}}

relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        "U|k|omega"     0.7;
    }}
}}
""",
        encoding="utf-8",
    )
