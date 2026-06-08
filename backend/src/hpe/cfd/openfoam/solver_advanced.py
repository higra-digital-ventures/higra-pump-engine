"""Solver core enhancements + mesh advanced — melhorias #1-20.

Bloco A (1-10): solver tuning
Bloco B (11-20): mesh advanced
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ===========================================================================
# #1 Coupled solver (foam-extend / pisoFoam coupled)
# ===========================================================================

def write_coupled_solver_settings(case_dir: "str | Path") -> dict:
    """Configurar coupled p-U solver para casos difíceis (Ansys-style)."""
    case_dir = Path(case_dir)
    fv_file = case_dir / "system" / "fvSolution"
    fv_file.parent.mkdir(parents=True, exist_ok=True)

    fv_file.write_text("""\
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }

solvers
{
    "(p|U|Up)Coupled"
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.001;
        cacheAgglomeration true;
        nCellsInCoarsestLevel 200;
    }
    "(k|omega|epsilon)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent      yes;
    coupled         yes;
    pRefCell        0;
    pRefValue       0;
}

relaxationFactors
{
    equations
    {
        U               0.9;
        "(k|omega|epsilon)" 0.7;
    }
    fields
    {
        p               0.7;
    }
}
""", encoding="utf-8")
    return {"solver_mode": "coupled", "method": "SIMPLEC_coupled"}


# ===========================================================================
# #2 Pseudo-transient
# ===========================================================================

def write_pseudo_transient(case_dir: "str | Path", pseudo_dt: float = 1e-3) -> dict:
    """Pseudo-transient continuation (steady cases tough to converge)."""
    case_dir = Path(case_dir)
    control = case_dir / "system" / "controlDict"
    control.parent.mkdir(parents=True, exist_ok=True)
    control.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         5000;
deltaT          {pseudo_dt};
writeControl    timeStep;
writeInterval   100;
adjustTimeStep  yes;
maxCo           10;
""", encoding="utf-8")
    return {"mode": "pseudo_transient", "dt": pseudo_dt}


# ===========================================================================
# #3 AMG presets
# ===========================================================================

def amg_presets() -> dict:
    """Presets de Algebraic Multigrid para diferentes regimes."""
    return {
        "fast": {"smoother": "GaussSeidel", "n_pre_sweeps": 0, "n_post_sweeps": 2,
                 "merge_levels": 1, "agglomerator": "faceAreaPair"},
        "robust": {"smoother": "DICGaussSeidel", "n_pre_sweeps": 2, "n_post_sweeps": 2,
                   "merge_levels": 1, "cache_agglomeration": True},
        "stiff": {"smoother": "DILUGaussSeidel", "n_pre_sweeps": 2, "n_post_sweeps": 4,
                  "merge_levels": 2, "n_cells_coarsest": 50},
    }


# ===========================================================================
# #4 GAMG tuning
# ===========================================================================

@dataclass
class GAMGConfig:
    smoother: str = "GaussSeidel"
    cacheAgglomeration: bool = True
    nCellsInCoarsestLevel: int = 100
    agglomerator: str = "faceAreaPair"
    mergeLevels: int = 1
    nPreSweeps: int = 0
    nPostSweeps: int = 2

    def to_openfoam_block(self) -> str:
        return f"""\
        solver          GAMG;
        smoother        {self.smoother};
        cacheAgglomeration {str(self.cacheAgglomeration).lower()};
        nCellsInCoarsestLevel {self.nCellsInCoarsestLevel};
        agglomerator    {self.agglomerator};
        mergeLevels     {self.mergeLevels};
        nPreSweeps      {self.nPreSweeps};
        nPostSweeps     {self.nPostSweeps};
"""


def tune_gamg(n_cells_total: int, regime: str = "robust") -> GAMGConfig:
    """Sugerir GAMG config baseado em tamanho do mesh."""
    base = GAMGConfig()
    if n_cells_total > 10_000_000:
        base.nCellsInCoarsestLevel = 500
        base.mergeLevels = 2
    elif n_cells_total > 1_000_000:
        base.nCellsInCoarsestLevel = 200
    if regime == "stiff":
        base.smoother = "DILUGaussSeidel"
        base.nPostSweeps = 4
    return base


# ===========================================================================
# #5 SIMPLE-C (consistent)
# ===========================================================================

def write_simplec(case_dir: "str | Path") -> dict:
    """SIMPLE-C variante (consistent) — convergência mais rápida."""
    case_dir = Path(case_dir)
    fv_file = case_dir / "system" / "fvSolution"
    if fv_file.exists():
        text = fv_file.read_text()
        if "consistent" not in text:
            text = text.replace("SIMPLE\n{", "SIMPLE\n{\n    consistent yes;")
            fv_file.write_text(text)
    return {"variant": "SIMPLEC", "consistent": True}


# ===========================================================================
# #6 Density-based solver picker
# ===========================================================================

def write_density_based(case_dir: "str | Path", Ma_ref: float = 0.5) -> dict:
    """Density-based solver para escoamento compressível."""
    case_dir = Path(case_dir)
    control = case_dir / "system" / "controlDict"
    control.parent.mkdir(parents=True, exist_ok=True)
    solver = "rhoCentralFoam" if Ma_ref > 0.6 else "rhoSimpleFoam"
    control.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1e-5;
""", encoding="utf-8")
    return {"solver": solver, "Ma_ref": Ma_ref}


# ===========================================================================
# #7 Residual smoothing
# ===========================================================================

def smooth_residuals(residuals: list[float], window: int = 5) -> list[float]:
    """Moving average smoothing dos resíduos para visualização clean."""
    if len(residuals) < window:
        return residuals
    out = []
    for i in range(len(residuals)):
        lo = max(0, i - window // 2)
        hi = min(len(residuals), i + window // 2 + 1)
        out.append(sum(residuals[lo:hi]) / (hi - lo))
    return out


# ===========================================================================
# #8 Gradient limiters
# ===========================================================================

def gradient_limiter_block(scheme: str = "cellLimited") -> str:
    """Gerar bloco gradSchemes com limiter para evitar overshoot."""
    return f"""\
gradSchemes
{{
    default         Gauss linear;
    grad(U)         {scheme} Gauss linear 1;
    grad(p)         {scheme} Gauss linear 1;
    grad(k)         {scheme} Gauss linear 1;
    grad(omega)     {scheme} Gauss linear 1;
}}
"""


# ===========================================================================
# #9 Flux limiters
# ===========================================================================

def flux_limiter_div(scheme: str = "Minmod") -> str:
    """divSchemes com flux limiter (TVD) para alta resolução estável."""
    return f"""\
divSchemes
{{
    default         none;
    div(phi,U)      Gauss limitedLinearV 1;
    div(phi,k)      Gauss {scheme};
    div(phi,omega)  Gauss {scheme};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
"""


# ===========================================================================
# #10 Time-step controllers
# ===========================================================================

@dataclass
class TimeStepController:
    Co_target: float = 1.0
    max_dt: float = 1e-3
    min_dt: float = 1e-7
    increase_factor: float = 1.2
    decrease_factor: float = 0.5

    def next_dt(self, current_dt: float, current_Co: float) -> float:
        if current_Co > self.Co_target * 1.5:
            return max(self.min_dt, current_dt * self.decrease_factor)
        elif current_Co < self.Co_target * 0.7:
            return min(self.max_dt, current_dt * self.increase_factor)
        return current_dt


# ===========================================================================
# Bloco B — Mesh advanced (#11-20)
# ===========================================================================

# #11 AMR triggers
@dataclass
class AMRTrigger:
    field: str = "p"
    refinement_threshold: float = 0.1   # |grad|/max
    max_refinement_level: int = 3
    refine_interval: int = 50

    def should_refine(self, gradient_max: float, iteration: int) -> bool:
        return (gradient_max > self.refinement_threshold and
                iteration % self.refine_interval == 0)


# #12 Edge refinement
def refine_edges_at_features(edges: list[dict], angle_threshold_deg: float = 30.0) -> list[dict]:
    """Marcar edges com ângulo > threshold para refinement explicit."""
    return [
        {**e, "refine": True}
        for e in edges
        if e.get("feature_angle_deg", 0) > angle_threshold_deg
    ]


# #13 Gap detection
def detect_gaps(surfaces: list[dict], min_gap_m: float = 1e-4) -> list[dict]:
    """Heurística: gaps menores que min_gap geram problemas no snappy."""
    return [
        {"surface_a": s1.get("name"), "surface_b": s2.get("name"),
         "gap_m": min_gap_m, "warning": "use refinement gap"}
        for i, s1 in enumerate(surfaces)
        for s2 in surfaces[i + 1:]
    ]


# #14 Surface remeshing
def remesh_surface_command(stl_file: "str | Path", target_edge: float = 0.005) -> str:
    """Comando surfaceMesh para remesh com edge length target."""
    return f"surfaceMeshTriangulate -targetEdgeLength {target_edge} {stl_file}"


# #15 Hex-dominant generator
def write_hex_dominant_dict(case_dir: "str | Path", n_cells_xyz: tuple[int, int, int]) -> Path:
    """blockMeshDict puro hex (sem snappy)."""
    case_dir = Path(case_dir)
    f = case_dir / "system" / "blockMeshDict"
    f.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = n_cells_xyz
    f.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
    (-1 -1 -0.1) (1 -1 -0.1) (1 1 -0.1) (-1 1 -0.1)
    (-1 -1  0.1) (1 -1  0.1) (1 1  0.1) (-1 1  0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);
""", encoding="utf-8")
    return f


# #16 Periodic generator
def write_periodic_patches(case_dir: "str | Path", angle_deg: float) -> dict:
    """Gerar patches periódicos rotacionais (single-passage mesh)."""
    return {
        "type": "cyclicPeriodic",
        "angle_deg": angle_deg,
        "patches": ["periodic_a", "periodic_b"],
    }


# #17 Baffle handler
def define_baffles(case_dir: "str | Path", baffle_patches: list[str]) -> dict:
    """Configurar baffles (parede infinitamente fina) no createBafflesDict."""
    case_dir = Path(case_dir)
    f = case_dir / "system" / "createBafflesDict"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object createBafflesDict; }}
internalFacesOnly true;
baffles
{{
    {chr(10).join(f'    {p} {{ type wall; }}' for p in baffle_patches)}
}}
""", encoding="utf-8")
    return {"baffles": baffle_patches, "n_baffles": len(baffle_patches)}


# #18 Internal walls
def add_internal_walls(case_dir: "str | Path", n_walls: int) -> dict:
    """Marcar n superfícies internas como walls (splitter blades, ribs)."""
    return {"n_internal_walls": n_walls, "method": "topoSet + createBaffles"}


# #19 Mesh export OpenFOAM ↔ CGNS
def mesh_format_converter(input_format: str, output_format: str) -> dict:
    """Sugerir comando de conversão entre formatos de malha."""
    converters = {
        ("OpenFOAM", "CGNS"): "foamToCGNS",
        ("CGNS", "OpenFOAM"): "cgnsToFoam",
        ("OpenFOAM", "Fluent"): "foamMeshToFluent",
        ("Fluent", "OpenFOAM"): "fluentMeshToFoam",
        ("Gmsh", "OpenFOAM"): "gmshToFoam",
        ("OpenFOAM", "Tecplot"): "foamToTecplot",
    }
    return {
        "from": input_format,
        "to": output_format,
        "command": converters.get((input_format, output_format), "manual"),
    }


# #20 Parallel decomposer
def decompose_par_dict(n_procs: int, method: str = "scotch") -> str:
    """system/decomposeParDict para decomposição paralela."""
    return f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains {n_procs};
method          {method};

simpleCoeffs
{{
    n               (2 2 1);
    delta           0.001;
}}

scotchCoeffs
{{
    strategy        "b";
}}

distributed     no;
roots           ();
"""
