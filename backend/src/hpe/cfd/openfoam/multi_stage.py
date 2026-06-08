"""Bombas multi-estágio — encadeamento rotor + difusor/voluta N vezes.

Fase 20.2 — habilita análise de bombas centrífugas de múltiplos estágios
(típicas de alta pressão: pump stacks de óleo&gás, boiler feed).

Estratégia:
  - N estágios idênticos ou distintos encadeados
  - Cada estágio = (rotor MRF) + (difusor/voluta estacionário)
  - Mixing plane interfaces entre estágios (média circunferencial
    no mapeamento rotor_out → stator_in, unwind)
  - BC global: inlet no estágio 1, outlet no estágio N

Usage
-----
    from hpe.cfd.openfoam.multi_stage import build_multistage_case, StageConfig

    stages = [StageConfig(sizing=s1), StageConfig(sizing=s2)]
    case = build_multistage_case(stages, output_dir)
    print(case.n_stages, case.total_head)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuração de um único estágio."""
    sizing: object                   # SizingResult deste estágio
    n_blades: int = 6
    diffuser_type: str = "vaned"     # 'vaned' | 'vaneless' | 'volute'
    stage_id: int = 0


@dataclass
class MultiStageCase:
    """Caso multi-estágio montado."""
    case_dir: Path
    n_stages: int
    stages: list[StageConfig]
    total_head: float
    stage_dirs: list[Path] = field(default_factory=list)
    interface_patches: list[tuple[str, str]] = field(default_factory=list)
    solver: str = "simpleFoam"
    created: bool = False

    def to_dict(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "n_stages": self.n_stages,
            "total_head": round(self.total_head, 2),
            "solver": self.solver,
            "stage_dirs": [str(p) for p in self.stage_dirs],
            "interface_patches": [list(p) for p in self.interface_patches],
            "created": self.created,
        }


def build_multistage_case(
    stages: list[StageConfig],
    output_dir: "str | Path",
    turbulence_model: str = "kOmegaSST",
    n_procs: int = 8,
) -> MultiStageCase:
    """Montar caso multi-estágio com mixing planes.

    Parameters
    ----------
    stages : list[StageConfig]
        Lista de estágios (cada um com seu sizing).
    output_dir : Path
        Diretório raiz do caso multi-estágio.
    turbulence_model : str
    n_procs : int
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_stages = len(stages)
    if n_stages == 0:
        raise ValueError("At least 1 stage required")

    # ── Construir cada estágio em seu sub-diretório ────────────────────────
    stage_dirs: list[Path] = []
    total_head = 0.0
    from hpe.cfd.openfoam.case import build_openfoam_case

    for i, stg in enumerate(stages):
        stg.stage_id = i
        stage_dir = output_dir / f"stage_{i:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        log.info("Multi-stage: building stage %d at %s", i, stage_dir)
        build_openfoam_case(
            sizing=stg.sizing,
            output_dir=stage_dir,
            mesh_mode="snappy",
            turbulence_model=turbulence_model,
            n_procs=n_procs // n_stages,
        )

        total_head += float(getattr(stg.sizing, "H", 30.0))
        stage_dirs.append(stage_dir)

    # ── Interfaces mixing plane entre estágios ─────────────────────────────
    interfaces: list[tuple[str, str]] = []
    for i in range(n_stages - 1):
        out_patch = f"stage{i}_out"
        in_patch = f"stage{i + 1}_in"
        interfaces.append((out_patch, in_patch))

    # ── createPatchDict global com mixing planes ───────────────────────────
    create_patch = output_dir / "system" / "createPatchDict"
    create_patch.parent.mkdir(parents=True, exist_ok=True)
    create_patch.write_text(_mixing_plane_patch_dict(interfaces), encoding="utf-8")

    # ── Allrun script ──────────────────────────────────────────────────────
    (output_dir / "Allrun.multistage").write_text(
        _allrun_multistage(n_stages), encoding="utf-8"
    )

    case = MultiStageCase(
        case_dir=output_dir,
        n_stages=n_stages,
        stages=stages,
        total_head=total_head,
        stage_dirs=stage_dirs,
        interface_patches=interfaces,
        solver="simpleFoam",
        created=True,
    )
    log.info(
        "Multi-stage case: %d stages, total H=%.1f m, %d interfaces",
        n_stages, total_head, len(interfaces),
    )
    return case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mixing_plane_patch_dict(interfaces: list[tuple[str, str]]) -> str:
    """Gerar createPatchDict com mixing plane para cada interface."""
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      createPatchDict;",
        "}",
        "",
        "pointSync false;",
        "",
        "patches",
        "(",
    ]
    for out_p, in_p in interfaces:
        lines += [
            "    {",
            f"        name            {out_p};",
            "        patchInfo",
            "        {",
            "            type            mixingPlane;",
            f"            neighbourPatch  {in_p};",
            "            ribbonPatch",
            "            {",
            "                sweepAxis   Z;",
            "                stackAxis   R;",
            "                discretisation  bothPatches;",
            "            }",
            "        }",
            "        constructFrom   patches;",
            f"        patches         ({out_p}_original);",
            "    }",
            "    {",
            f"        name            {in_p};",
            "        patchInfo",
            "        {",
            "            type            mixingPlane;",
            f"            neighbourPatch  {out_p};",
            "        }",
            "        constructFrom   patches;",
            f"        patches         ({in_p}_original);",
            "    }",
        ]
    lines += [");", ""]
    return "\n".join(lines)


def _allrun_multistage(n_stages: int) -> str:
    stage_loop = "\n".join(
        f'cd stage_{i:02d} && blockMesh && snappyHexMesh -overwrite && cd ..'
        for i in range(n_stages)
    )
    return f"""\
#!/bin/bash
# Allrun multi-stage — Fase 20.2
set -e

# Build each stage mesh
{stage_loop}

# Merge all meshes
mergeMeshes -overwrite . stage_01
for i in $(seq -f "%02g" 2 $(({n_stages} - 1))); do
  mergeMeshes -overwrite . stage_$i
done

# Create mixing plane interfaces
createPatch -overwrite

# Decompose and run
decomposePar -force
mpirun -np $(nproc) simpleFoam -parallel
reconstructPar

echo "Multi-stage run complete. Total stages: {n_stages}"
"""
