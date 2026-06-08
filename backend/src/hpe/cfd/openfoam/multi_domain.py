"""Multi-domain rotor + voluta com interface AMI — Fase 17.1.

Monta um caso OpenFOAM com dois domínios acoplados:
  - rotor/  (MRF rotativo)
  - volute/ (estacionário)
  - Interface AMI (Arbitrary Mesh Interface = equivalente ao GGI do CFX)

Usage
-----
    from hpe.cfd.openfoam.multi_domain import build_multi_domain_case

    case = build_multi_domain_case(
        sizing=sizing_result,
        volute_geometry=volute,
        output_dir=Path("cfd_multi"),
        turbulence_model="kOmegaSST",
    )
    print(case.rotor_dir, case.volute_dir, case.interface_patches)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class MultiDomainCase:
    """Resultado da montagem de um caso multi-domínio."""
    case_dir: Path
    rotor_dir: Path
    volute_dir: Path
    interface_patches: dict[str, str]  # {"rotor_outlet": "volute_inlet"}
    turbulence_model: str
    n_procs: int
    created: bool = False

    def to_dict(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "rotor_dir": str(self.rotor_dir),
            "volute_dir": str(self.volute_dir),
            "interface_patches": self.interface_patches,
            "turbulence_model": self.turbulence_model,
            "n_procs": self.n_procs,
            "created": self.created,
        }


def build_multi_domain_case(
    sizing,
    volute_geometry=None,
    output_dir: "str | Path" = "cfd_multi",
    turbulence_model: str = "kOmegaSST",
    n_procs: int = 4,
    mesh_mode: str = "snappy",
) -> MultiDomainCase:
    """Montar caso CFD rotor+voluta acoplado via AMI.

    Usa `mergeMeshes` para combinar os dois domínios em um único mesh
    e cria patches `cyclicAMI` na interface rotor/voluta.
    """
    output_dir = Path(output_dir)
    rotor_dir  = output_dir / "rotor"
    volute_dir = output_dir / "volute"
    rotor_dir.mkdir(parents=True, exist_ok=True)
    volute_dir.mkdir(parents=True, exist_ok=True)

    # ── Construir o caso do rotor com as BCs adaptadas ──────────────────────
    from hpe.cfd.openfoam.case import build_openfoam_case

    log.info("Multi-domain: building rotor case at %s", rotor_dir)
    build_openfoam_case(
        sizing=sizing,
        output_dir=rotor_dir,
        mesh_mode=mesh_mode,
        turbulence_model=turbulence_model,
        n_procs=n_procs,
    )
    _patch_rotor_outlet_to_ami(rotor_dir, patch_name="rotor_to_volute")

    # ── Construir o caso da voluta (estacionário) ───────────────────────────
    log.info("Multi-domain: building volute case at %s", volute_dir)
    _build_volute_case(
        sizing=sizing,
        volute_geometry=volute_geometry,
        output_dir=volute_dir,
        turbulence_model=turbulence_model,
    )

    # ── Escrever topoSetDict / createPatchDict para merge + AMI ─────────────
    merge_dict = output_dir / "system" / "mergeMeshDict"
    merge_dict.parent.mkdir(parents=True, exist_ok=True)
    merge_dict.write_text(_merge_mesh_dict(rotor_dir, volute_dir), encoding="utf-8")

    create_patch_dict = output_dir / "system" / "createPatchDict"
    create_patch_dict.write_text(_create_patch_dict_ami(), encoding="utf-8")

    interface_patches = {
        "rotor_to_volute": "volute_to_rotor",
    }

    case = MultiDomainCase(
        case_dir=output_dir,
        rotor_dir=rotor_dir,
        volute_dir=volute_dir,
        interface_patches=interface_patches,
        turbulence_model=turbulence_model,
        n_procs=n_procs,
        created=True,
    )

    # Script de montagem que o usuário executa
    (output_dir / "Allrun.multi_domain").write_text(
        _allrun_script(rotor_dir.name, volute_dir.name), encoding="utf-8"
    )
    log.info("Multi-domain case ready at %s", output_dir)
    return case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_rotor_outlet_to_ami(rotor_dir: Path, patch_name: str) -> None:
    """Ajustar BC do outlet do rotor para cyclicAMI."""
    p_file = rotor_dir / "0" / "p"
    if not p_file.exists():
        return
    text = p_file.read_text(encoding="utf-8")
    # Substituir BC do outlet para cyclicAMI type
    text = text.replace(
        "    outlet\n    {\n        type            fixedValue;",
        f"    {patch_name}\n    {{\n        type            cyclicAMI;",
    )
    p_file.write_text(text, encoding="utf-8")


def _build_volute_case(
    sizing,
    volute_geometry,
    output_dir: Path,
    turbulence_model: str,
) -> None:
    """Gerar caso estático da voluta (sem MRF, apenas dissipação)."""
    (output_dir / "0").mkdir(parents=True, exist_ok=True)
    (output_dir / "constant").mkdir(parents=True, exist_ok=True)
    (output_dir / "system").mkdir(parents=True, exist_ok=True)

    # controlDict mínimo
    (output_dir / "system" / "controlDict").write_text(
        _minimal_control_dict("simpleFoam"), encoding="utf-8"
    )
    # turbulenceProperties
    (output_dir / "constant" / "turbulenceProperties").write_text(
        _turbulence_props(turbulence_model), encoding="utf-8"
    )
    # transportProperties
    (output_dir / "constant" / "transportProperties").write_text(
        "transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] 1e-6;\n",
        encoding="utf-8",
    )
    # Placeholder p/U BCs
    (output_dir / "0" / "p").write_text(_placeholder_p_field(), encoding="utf-8")
    (output_dir / "0" / "U").write_text(_placeholder_u_field(), encoding="utf-8")


def _merge_mesh_dict(rotor: Path, volute: Path) -> str:
    return f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      mergeMeshDict;
}}

// Fundir rotor + voluta em um único mesh de caso multi-domínio
masterCase  "{rotor.resolve()}";
addCase     "{volute.resolve()}";
"""


def _create_patch_dict_ami() -> str:
    return """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}

pointSync false;

patches
(
    {
        name            rotor_to_volute;
        patchInfo
        {
            type        cyclicAMI;
            neighbourPatch volute_to_rotor;
            transform   noOrdering;
        }
        constructFrom   patches;
        patches         (outlet);
    }
    {
        name            volute_to_rotor;
        patchInfo
        {
            type        cyclicAMI;
            neighbourPatch rotor_to_volute;
            transform   noOrdering;
        }
        constructFrom   patches;
        patches         (volute_inlet);
    }
);
"""


def _allrun_script(rotor_name: str, volute_name: str) -> str:
    return f"""\
#!/bin/bash
# Allrun — multi-domain rotor + voluta (Fase 17.1)
set -e

cd {rotor_name} && blockMesh && snappyHexMesh -overwrite && cd ..
cd {volute_name} && blockMesh && snappyHexMesh -overwrite && cd ..

# Merge meshes
mergeMeshes -overwrite . {volute_name}

# Apply cyclicAMI patches
createPatch -overwrite

# Run solver
decomposePar -force
mpirun -np $(nproc) simpleFoam -parallel
reconstructPar

echo "Multi-domain run complete."
"""


def _minimal_control_dict(application: str) -> str:
    return f"""\
FoamFile
{{ version 2.0; format ascii; class dictionary; object controlDict; }}
application     {application};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         500;
deltaT          1;
writeControl    timeStep;
writeInterval   50;
purgeWrite      0;
runTimeModifiable true;
"""


def _turbulence_props(model: str) -> str:
    return f"""\
FoamFile
{{ version 2.0; format ascii; class dictionary; object turbulenceProperties; }}
simulationType  RAS;
RAS
{{
    RASModel        {model};
    turbulence      on;
    printCoeffs     on;
}}
"""


def _placeholder_p_field() -> str:
    return """\
FoamFile
{ version 2.0; format ascii; class volScalarField; object p; }
dimensions  [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    volute_inlet  { type fixedValue; value uniform 0; }
    volute_outlet { type zeroGradient; }
    walls         { type zeroGradient; }
    volute_to_rotor { type cyclicAMI; }
}
"""


def _placeholder_u_field() -> str:
    return """\
FoamFile
{ version 2.0; format ascii; class volVectorField; object U; }
dimensions  [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    volute_inlet  { type zeroGradient; }
    volute_outlet { type fixedValue; value uniform (0 0 0); }
    walls         { type noSlip; }
    volute_to_rotor { type cyclicAMI; }
}
"""
