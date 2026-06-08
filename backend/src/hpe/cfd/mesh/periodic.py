"""Periodic boundary condition setup for single-passage turbomachinery CFD.

Generates OpenFOAM files needed to configure cyclic (periodic) boundaries
for a single blade passage simulation:

  - system/createPatchDict   used with `createPatch` utility
  - Boundary entry patches in 0/ fields

The passage spans one blade pitch = 2*pi / Z (radians).
The two periodic faces are called periodicLow (θ=0) and periodicHigh (θ=pitch).

References:
    - OpenFOAM User Guide §5.2 — Cyclic boundaries
    - Gulich (2014) §8.1 — Single-passage CFD setup
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PeriodicConfig:
    """Configuration for periodic blade passage boundaries.

    Attributes
    ----------
    blade_count : int
        Number of blades Z. Pitch = 2*pi/Z [rad].
    patch_low_name : str
        Name of the low-theta periodic patch (default 'periodicLow').
    patch_high_name : str
        Name of the high-theta periodic patch (default 'periodicHigh').
    rotation_axis : tuple[float, float, float]
        Rotation axis vector. Default (0,0,1) = Z-axis for centrifugal.
    rotation_origin : tuple[float, float, float]
        Origin for the rotation transform.
    """
    blade_count: int
    patch_low_name: str = "periodicLow"
    patch_high_name: str = "periodicHigh"
    rotation_axis: tuple = (0.0, 0.0, 1.0)
    rotation_origin: tuple = (0.0, 0.0, 0.0)

    @property
    def pitch_rad(self) -> float:
        """Blade pitch in radians."""
        return 2.0 * math.pi / self.blade_count

    @property
    def pitch_deg(self) -> float:
        """Blade pitch in degrees."""
        return math.degrees(self.pitch_rad)


# Default field boundary condition templates
_CYCLIC_BC = """\
    {name}
    {{
        type        cyclic;
        neighbourPatch {neighbour};
    }}"""

_CYCLIC_AMI_BC = """\
    {name}
    {{
        type            cyclicAMI;
        neighbourPatch  {neighbour};
        matchTolerance  0.0001;
        transform       rotational;
        rotationAxis    ({ax} {ay} {az});
        rotationCentre  ({ox} {oy} {oz});
        rotationAngle   {angle_deg};
    }}"""


def write_create_patch_dict(
    case_dir: Path,
    config: PeriodicConfig,
    use_ami: bool = False,
) -> Path:
    """Write system/createPatchDict for cyclic periodic boundaries.

    This file is used with the OpenFOAM `createPatch` utility to link
    periodicLow and periodicHigh into a cyclic pair.

    Args:
        case_dir: OpenFOAM case root directory.
        config: Periodic boundary configuration.
        use_ami: If True, write cyclicAMI (for non-conformal matching).
            Default False uses direct cyclic (requires exact face matching).

    Returns:
        Path to the written file (system/createPatchDict).
    """
    case_dir = Path(case_dir)
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    out = system_dir / "createPatchDict"

    patch_type = "cyclicAMI" if use_ami else "cyclic"
    ax, ay, az = config.rotation_axis
    ox, oy, oz = config.rotation_origin

    content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}}

pointSync false;

patches
(
    // Low periodic patch — theta = 0
    {{
        name            {config.patch_low_name};
        patchInfo
        {{
            type            {patch_type};
            neighbourPatch  {config.patch_high_name};
            matchTolerance  0.0001;
            transform       rotational;
            rotationAxis    ({ax} {ay} {az});
            rotationCentre  ({ox} {oy} {oz});
            rotationAngle   {-config.pitch_deg:.6f};
        }}
        constructFrom   patches;
        patches         ( periodicLow_base );
    }}

    // High periodic patch — theta = pitch
    {{
        name            {config.patch_high_name};
        patchInfo
        {{
            type            {patch_type};
            neighbourPatch  {config.patch_low_name};
            matchTolerance  0.0001;
            transform       rotational;
            rotationAxis    ({ax} {ay} {az});
            rotationCentre  ({ox} {oy} {oz});
            rotationAngle   {config.pitch_deg:.6f};
        }}
        constructFrom   patches;
        patches         ( periodicHigh_base );
    }}
);
"""
    out.write_text(content)
    return out


def write_periodic_boundary_conditions(
    case_dir: Path,
    config: PeriodicConfig,
    field_names: Optional[list] = None,
) -> dict:
    """Write cyclic boundary entries into 0/ field files.

    Appends cyclic boundary blocks for periodicLow and periodicHigh
    to the existing field initial condition files.

    Args:
        case_dir: OpenFOAM case root directory.
        config: Periodic boundary configuration.
        field_names: Field files to patch. Defaults to U, p, k, epsilon, nut, omega.

    Returns:
        Dict mapping field_name → path for each modified/created file.
    """
    if field_names is None:
        field_names = ["U", "p", "k", "epsilon", "nut", "omega"]

    case_dir = Path(case_dir)
    zero_dir = case_dir / "0"
    zero_dir.mkdir(parents=True, exist_ok=True)

    ax, ay, az = config.rotation_axis
    ox, oy, oz = config.rotation_origin

    bc_block = f"""
    {config.patch_low_name}
    {{
        type        cyclic;
        neighbourPatch {config.patch_high_name};
    }}

    {config.patch_high_name}
    {{
        type        cyclic;
        neighbourPatch {config.patch_low_name};
    }}"""

    results = {}
    for field in field_names:
        fpath = zero_dir / field
        if fpath.exists():
            text = fpath.read_text()
            # Inject periodic patches before the closing brace of boundaryField
            if "boundaryField" in text and config.patch_low_name not in text:
                insert_pos = text.rfind("}")
                if insert_pos > 0:
                    text = text[:insert_pos] + bc_block + "\n}" + text[insert_pos + 1:]
                    fpath.write_text(text)
            results[field] = fpath

    return results


def get_periodic_blockmesh_bc_entry(config: PeriodicConfig) -> str:
    """Return the blockMeshDict boundary entry string for periodic patches.

    Use this in the blockMeshDict `boundary` section to declare
    periodicLow and periodicHigh as cyclic patches.

    Args:
        config: Periodic boundary configuration.

    Returns:
        String to embed in blockMeshDict boundary section.
    """
    ax, ay, az = config.rotation_axis
    ox, oy, oz = config.rotation_origin
    return f"""
    {config.patch_low_name}
    {{
        type    cyclic;
        neighbourPatch {config.patch_high_name};
        transform rotational;
        rotationAxis  ({ax} {ay} {az});
        rotationCentre ({ox} {oy} {oz});
        faces ( $periodicLow_faces );
    }}

    {config.patch_high_name}
    {{
        type    cyclic;
        neighbourPatch {config.patch_low_name};
        transform rotational;
        rotationAxis  ({ax} {ay} {az});
        rotationCentre ({ox} {oy} {oz});
        faces ( $periodicHigh_faces );
    }}"""
