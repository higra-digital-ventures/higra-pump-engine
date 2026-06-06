"""Implicit (SDF/voxel) geometry — the free/generative backend (M3 prototype).

Pure-numpy signed-distance-field primitives and a sampler. This is a feasibility
prototype in the spirit of LEAP 71's PicoGK (which is voxel/SDF based): geometry
is defined by a scalar field rather than a B-rep, which suits complex internal
channels and additive manufacturing.

Optional STL export uses scikit-image marching cubes when available; otherwise
the voxel field + statistics are still produced.
"""

from hpe.geometry.implicit.sdf import (  # noqa: F401
    sphere, capped_cylinder_z, box, rotate_z,
    union, intersection, subtract, smooth_union,
)
from hpe.geometry.implicit.field import ScalarField, sample  # noqa: F401
