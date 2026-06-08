"""OpenFOAM solver configuration files — Fase 2 CFD Pipeline.

Funções para gerar system/controlDict, system/fvSchemes, system/fvSolution,
constant/transportProperties, constant/turbulenceProperties,
constant/MRFProperties e o script run.sh.

Todos os arquivos gerados são compatíveis com MRFSimpleFoam (regime permanente)
para simulação de bombas centrífugas.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Cabeçalho padrão OpenFOAM
# ---------------------------------------------------------------------------

_FOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _write(path: Path, content: str) -> Path:
    """Escrever conteúdo em path, criando diretórios pai se necessário."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# system/controlDict
# ---------------------------------------------------------------------------


def write_control_dict(case_dir: Path, n_iter: int = 500) -> Path:
    """Escrever system/controlDict para MRFSimpleFoam (steady-state).

    Parameters
    ----------
    case_dir : Path
        Raiz do caso OpenFOAM.
    n_iter : int
        Número máximo de iterações (endTime em regime permanente).

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    content = _FOAM_HEADER.format(cls="dictionary", obj="controlDict")
    content += f"""
application     MRFSimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {n_iter};

deltaT          1;

writeControl    timeStep;
writeInterval   50;

purgeWrite      3;
writeFormat     ascii;
writePrecision  8;

runTimeModifiable true;

functions
{{
    residuals
    {{
        type            solverInfo;
        libs            ("libutilityFunctionObjects.so");
        fields          (U p k epsilon);
        writeResidualFields no;
    }}

    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         (rotorWalls);
        rho             rhoInf;
        rhoInf          998.2;
        CofR            (0 0 0);
        writeControl    timeStep;
        writeInterval   10;
    }}

    flowRateInlet
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        fields          (phi);
        operation       sum;
        regionType      patch;
        name            inlet;
        writeControl    timeStep;
        writeInterval   10;
    }}

    pressureAvgInlet
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        fields          (p);
        operation       areaAverage;
        regionType      patch;
        name            inlet;
        writeControl    timeStep;
        writeInterval   10;
    }}

    pressureAvgOutlet
    {{
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        fields          (p);
        operation       areaAverage;
        regionType      patch;
        name            outlet;
        writeControl    timeStep;
        writeInterval   10;
    }}
}}
"""
    return _write(case_dir / "system" / "controlDict", content)


# ---------------------------------------------------------------------------
# system/fvSchemes
# ---------------------------------------------------------------------------


def write_fv_schemes(case_dir: Path) -> Path:
    """Escrever system/fvSchemes para turbomáquinas (regime permanente).

    Usa esquemas upwind limitados para estabilidade com fluxos rotacionais.

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    content = _FOAM_HEADER.format(cls="dictionary", obj="fvSchemes")
    content += """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(k)         cellLimited Gauss linear 1;
    grad(epsilon)   cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;

    div(phi,U)      bounded Gauss linearUpwindV grad(U);
    div(phi,k)      bounded Gauss linearUpwind grad(k);
    div(phi,epsilon) bounded Gauss linearUpwind grad(epsilon);
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

fluxRequired
{
    default         no;
    p               ;
}
"""
    return _write(case_dir / "system" / "fvSchemes", content)


# ---------------------------------------------------------------------------
# system/fvSolution
# ---------------------------------------------------------------------------


def write_fv_solution(case_dir: Path, turbulence_model: str = "kEpsilon") -> Path:
    """Escrever system/fvSolution com solvers SIMPLE para bomba.

    Usa GAMG para pressão e PBiCGStab para velocidade/turbulência.

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    content = _FOAM_HEADER.format(cls="dictionary", obj="fvSolution")
    content += """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }

    U
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.1;
    }

    k
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.1;
    }

    epsilon
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.1;
    }

    omega
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-7;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors  2;
    consistent                yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        k               1e-4;
        epsilon         1e-4;
        omega           1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.5;
        epsilon         0.5;
        omega           0.5;
    }
}
"""
    return _write(case_dir / "system" / "fvSolution", content)


# ---------------------------------------------------------------------------
# constant/transportProperties
# ---------------------------------------------------------------------------


def write_transport_properties(
    case_dir: Path,
    nu: float = 1e-6,
    rho: float = 998.2,
) -> Path:
    """Escrever constant/transportProperties (Newtoniano incompressível).

    Parameters
    ----------
    case_dir : Path
    nu : float
        Viscosidade cinemática [m²/s]. Default: água a 20°C.
    rho : float
        Massa específica [kg/m³]. Default: água a 20°C.

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    content = _FOAM_HEADER.format(cls="dictionary", obj="transportProperties")
    content += f"""
transportModel  Newtonian;

// Viscosidade cinemática [m²/s]
nu              {nu:.8e};

// Massa específica [kg/m³] (usada nas funções postProcessing)
rhoInf          {rho:.4f};
"""
    return _write(case_dir / "constant" / "transportProperties", content)


# ---------------------------------------------------------------------------
# constant/turbulenceProperties
# ---------------------------------------------------------------------------


def write_turbulence_properties(
    case_dir: Path,
    model: str = "kEpsilon",
) -> Path:
    """Escrever constant/turbulenceProperties (momentumTransport).

    Parameters
    ----------
    case_dir : Path
    model : str
        Modelo de turbulência: 'kEpsilon' | 'kOmegaSST' | 'realizableKE'.

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    content = _FOAM_HEADER.format(cls="dictionary", obj="turbulenceProperties")
    content += f"""
simulationType  RAS;

RAS
{{
    RASModel        {model};

    turbulence      on;
    printCoeffs     on;
}}
"""
    # OpenFOAM v8+ usa momentumTransport em vez de turbulenceProperties
    # Escrever ambos por compatibilidade
    _write(case_dir / "constant" / "turbulenceProperties", content)

    content2 = _FOAM_HEADER.format(cls="dictionary", obj="momentumTransport")
    content2 += f"""
simulationType  RAS;

RAS
{{
    model           {model};

    turbulence      on;
    printCoeffs     on;
}}
"""
    _write(case_dir / "constant" / "momentumTransport", content2)

    return case_dir / "constant" / "turbulenceProperties"


# ---------------------------------------------------------------------------
# constant/MRFProperties
# ---------------------------------------------------------------------------


def write_mrf_properties(
    case_dir: Path,
    omega_rad_s: float,
    zone_name: str = "rotatingZone",
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    non_rotating_patches: list[str] | None = None,
) -> Path:
    """Escrever constant/MRFProperties para Multiple Reference Frame.

    Parameters
    ----------
    case_dir : Path
    omega_rad_s : float
        Velocidade angular [rad/s]. Positivo = sentido anti-horário (eixo Z).
    zone_name : str
        Nome da zona de células rotativas (cellZone no polyMesh).
    axis : tuple
        Eixo de rotação (normalmente (0, 0, 1) para eixo Z).
    origin : tuple
        Origem do eixo de rotação.
    non_rotating_patches : list[str] | None
        Patches excluídos da rotação (inlet, outlet por padrão).

    Returns
    -------
    Path
        Caminho do arquivo gerado.
    """
    if non_rotating_patches is None:
        non_rotating_patches = ["inlet", "outlet", "statorWalls"]

    patches_str = "\n            ".join(non_rotating_patches)

    content = _FOAM_HEADER.format(cls="dictionary", obj="MRFProperties")
    content += f"""
MRF1
{{
    cellZone        {zone_name};

    active          yes;

    nonRotatingPatches
    (
        {patches_str}
    );

    origin          ({origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f});
    axis            ({axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f});
    omega           {omega_rad_s:.8f};  // [rad/s]
}}
"""
    return _write(case_dir / "constant" / "MRFProperties", content)


# ---------------------------------------------------------------------------
# run.sh
# ---------------------------------------------------------------------------


def write_run_script(case_dir: Path, n_procs: int = 4) -> Path:
    """Escrever run.sh com pipeline completo: blockMesh → snappyHexMesh →
    decomposePar → mpirun MRFSimpleFoam.

    Parameters
    ----------
    case_dir : Path
        Raiz do caso OpenFOAM.
    n_procs : int
        Número de processos MPI. Se 1, executa em modo serial.

    Returns
    -------
    Path
        Caminho do script gerado (com permissão de execução 0o755).
    """
    if n_procs > 1:
        solver_block = f"""\
echo "--- Step 4: decomposePar ---"
decomposePar -force

echo "--- Step 5: MRFSimpleFoam (parallel, {n_procs} procs) ---"
mpirun -np {n_procs} MRFSimpleFoam -parallel 2>&1 | tee log.MRFSimpleFoam

echo "--- Step 6: reconstructPar ---"
reconstructPar -latestTime
"""
    else:
        solver_block = """\
echo "--- Step 4: MRFSimpleFoam (serial) ---"
MRFSimpleFoam 2>&1 | tee log.MRFSimpleFoam
"""

    content = f"""#!/bin/bash
# HPE OpenFOAM run script — gerado automaticamente por hpe.cfd
# Altere n_procs, endTime e turbulence model conforme necessário.

set -euo pipefail

echo "=== HPE: Iniciando simulação OpenFOAM ==="
echo "    Case: $(pwd)"
echo "    Procs: {n_procs}"
date

# 1. Background mesh
echo "--- Step 1: blockMesh ---"
blockMesh 2>&1 | tee log.blockMesh

# 2. Feature extraction (opcional, silencioso se falhar)
echo "--- Step 2: surfaceFeatureExtract ---"
surfaceFeatureExtract 2>/dev/null || true

# 3. Refinamento snappyHexMesh
echo "--- Step 3: snappyHexMesh ---"
snappyHexMesh -overwrite 2>&1 | tee log.snappyHexMesh

{solver_block}

echo "=== HPE: Simulação concluída ==="
date
"""
    path = case_dir / "run.sh"
    _write(path, content)
    try:
        path.chmod(0o755)
    except Exception:
        pass  # Windows não suporta chmod
    return path
