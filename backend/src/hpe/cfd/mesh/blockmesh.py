"""BlockMesh generator — background mesh for snappyHexMesh.

Creates a cylindrical background mesh that envelops the pump geometry
with sufficient margin. snappyHexMesh then refines and snaps to the
actual STEP surface.
"""

from __future__ import annotations



def generate_blockmesh_dict(
    d2: float,
    b2: float,
    n_cells_radial: int = 20,
    n_cells_axial: int = 15,
    n_cells_tangential: int = 40,
    margin_factor: float = 2.0,
) -> str:
    """Generate blockMeshDict content for a cylindrical background mesh.

    The mesh is a rectangular box large enough to contain the pump
    geometry plus margins for inlet/outlet piping.

    Args:
        d2: Impeller outlet diameter [m].
        b2: Impeller outlet width [m].
        n_cells_radial: Cells in radial direction.
        n_cells_axial: Cells in axial direction.
        n_cells_tangential: Cells in tangential direction.
        margin_factor: Size multiplier beyond D2.

    Returns:
        blockMeshDict file content as string.
    """
    # Domain size (in meters, OpenFOAM uses meters)
    r_domain = d2 * margin_factor / 2.0
    z_min = -d2 * 0.5  # Below outlet plane
    z_max = d2 * 1.0   # Above inlet

    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
    ({-r_domain:.6f} {-r_domain:.6f} {z_min:.6f})
    ({ r_domain:.6f} {-r_domain:.6f} {z_min:.6f})
    ({ r_domain:.6f} { r_domain:.6f} {z_min:.6f})
    ({-r_domain:.6f} { r_domain:.6f} {z_min:.6f})
    ({-r_domain:.6f} {-r_domain:.6f} {z_max:.6f})
    ({ r_domain:.6f} {-r_domain:.6f} {z_max:.6f})
    ({ r_domain:.6f} { r_domain:.6f} {z_max:.6f})
    ({-r_domain:.6f} { r_domain:.6f} {z_max:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n_cells_tangential} {n_cells_radial} {n_cells_axial}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (0 3 2 1)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (3 0 4 7)
        );
    }}
);
"""
