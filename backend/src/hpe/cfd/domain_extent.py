"""Domain extent and flaring control for CFD mesh generation.

Controls the computational domain extension around the impeller
(inlet upstream / outlet downstream) with optional hub/shroud flaring
to avoid mesh distortion and improve CFD convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MeridionalLine:
    """A meridional contour line represented as (r, z) arrays."""

    r: NDArray[np.float64]
    z: NDArray[np.float64]

    def as_dict(self) -> dict[str, list[float]]:
        return {"r": self.r.tolist(), "z": self.z.tolist()}


@dataclass
class ExtendedDomain:
    """Full computational domain with inlet/blade/outlet sections for hub and shroud."""

    inlet_hub: MeridionalLine
    inlet_shroud: MeridionalLine
    blade_hub: MeridionalLine
    blade_shroud: MeridionalLine
    outlet_hub: MeridionalLine
    outlet_shroud: MeridionalLine

    def as_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "inlet_hub": self.inlet_hub.as_dict(),
            "inlet_shroud": self.inlet_shroud.as_dict(),
            "blade_hub": self.blade_hub.as_dict(),
            "blade_shroud": self.blade_shroud.as_dict(),
            "outlet_hub": self.outlet_hub.as_dict(),
            "outlet_shroud": self.outlet_shroud.as_dict(),
        }


FlaringType = Literal["linear", "parabolic", "exponential"]


# ---------------------------------------------------------------------------
# Flaring helpers
# ---------------------------------------------------------------------------

def _flaring_profile(
    n_pts: int,
    flaring_ratio: float,
    flaring_type: FlaringType,
) -> NDArray[np.float64]:
    """Return a 1-D array of multiplicative factors along the extension.

    Index 0 corresponds to the blade edge (factor=1.0), index n_pts-1
    corresponds to the far-field boundary (factor=flaring_ratio).
    """
    t = np.linspace(0.0, 1.0, n_pts)
    delta = flaring_ratio - 1.0

    if flaring_type == "linear":
        return 1.0 + delta * t
    elif flaring_type == "parabolic":
        return 1.0 + delta * t ** 2
    elif flaring_type == "exponential":
        if abs(delta) < 1e-12:
            return np.ones(n_pts)
        return 1.0 + delta * (np.exp(t) - 1.0) / (math.e - 1.0)
    else:
        raise ValueError(f"Unknown flaring_type: {flaring_type!r}")


# ---------------------------------------------------------------------------
# DomainExtent class
# ---------------------------------------------------------------------------

@dataclass
class DomainExtent:
    """Controls the computational domain extent and flaring around the impeller.

    Parameters
    ----------
    inlet_extension : float
        Upstream extension as multiples of D1.
    outlet_extension : float
        Downstream extension as multiples of D2.
    inlet_hub_ratio : float
        Hub flaring ratio at inlet (1.0 = straight, >1 = expanding).
    inlet_shroud_ratio : float
        Shroud flaring ratio at inlet.
    outlet_hub_ratio : float
        Hub flaring ratio at outlet.
    outlet_shroud_ratio : float
        Shroud flaring ratio at outlet.
    flaring_type : FlaringType
        Transition shape: 'linear', 'parabolic', or 'exponential'.
    n_extension_pts : int
        Number of points used for each extension section.
    """

    inlet_extension: float = 3.0
    outlet_extension: float = 5.0
    inlet_hub_ratio: float = 1.0
    inlet_shroud_ratio: float = 1.0
    outlet_hub_ratio: float = 1.0
    outlet_shroud_ratio: float = 1.0
    flaring_type: FlaringType = "linear"
    n_extension_pts: int = 30

    # -----------------------------------------------------------------
    # Main generation
    # -----------------------------------------------------------------

    def generate_domain(
        self,
        hub_rz: NDArray[np.float64],
        shroud_rz: NDArray[np.float64],
        d1: float,
        d2: float,
    ) -> ExtendedDomain:
        """Generate the extended computational domain.

        Parameters
        ----------
        hub_rz : ndarray, shape (N, 2)
            Hub meridional contour — columns (r, z).
        shroud_rz : ndarray, shape (M, 2)
            Shroud meridional contour — columns (r, z).
        d1 : float
            Inlet diameter [m].
        d2 : float
            Outlet diameter [m].

        Returns
        -------
        ExtendedDomain
        """
        hub_rz = np.asarray(hub_rz, dtype=np.float64)
        shroud_rz = np.asarray(shroud_rz, dtype=np.float64)

        # --- blade section (pass-through) ---
        blade_hub = MeridionalLine(hub_rz[:, 0].copy(), hub_rz[:, 1].copy())
        blade_shroud = MeridionalLine(shroud_rz[:, 0].copy(), shroud_rz[:, 1].copy())

        # --- inlet extension (upstream from LE) ---
        inlet_length = self.inlet_extension * d1
        inlet_hub = self._extend_inlet(
            hub_rz, inlet_length, self.inlet_hub_ratio,
        )
        inlet_shroud = self._extend_inlet(
            shroud_rz, inlet_length, self.inlet_shroud_ratio,
        )

        # --- outlet extension (downstream from TE) ---
        outlet_length = self.outlet_extension * d2
        outlet_hub = self._extend_outlet(
            hub_rz, outlet_length, self.outlet_hub_ratio,
        )
        outlet_shroud = self._extend_outlet(
            shroud_rz, outlet_length, self.outlet_shroud_ratio,
        )

        return ExtendedDomain(
            inlet_hub=inlet_hub,
            inlet_shroud=inlet_shroud,
            blade_hub=blade_hub,
            blade_shroud=blade_shroud,
            outlet_hub=outlet_hub,
            outlet_shroud=outlet_shroud,
        )

    # -----------------------------------------------------------------
    # Multi-grid
    # -----------------------------------------------------------------

    def generate_multi_grid(
        self,
        domain: ExtendedDomain,
        levels: int = 3,
    ) -> list[ExtendedDomain]:
        """Create coarse → fine domain levels for grid convergence studies.

        Each successive level doubles the number of points in every section.
        Level 0 is the coarsest (half the original points).
        """
        grids: list[ExtendedDomain] = []
        for lvl in range(levels):
            factor = 2 ** lvl
            grids.append(self._resample_domain(domain, factor))
        return grids

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate_domain(
        self,
        domain: ExtendedDomain,
        max_aspect_ratio: float = 50.0,
        max_skew_deg: float = 75.0,
    ) -> dict[str, list[str]]:
        """Run basic quality checks on the generated domain.

        Returns a dict with 'errors' and 'warnings' lists.
        """
        errors: list[str] = []
        warnings: list[str] = []

        for name, line in [
            ("inlet_hub", domain.inlet_hub),
            ("inlet_shroud", domain.inlet_shroud),
            ("blade_hub", domain.blade_hub),
            ("blade_shroud", domain.blade_shroud),
            ("outlet_hub", domain.outlet_hub),
            ("outlet_shroud", domain.outlet_shroud),
        ]:
            # Check for negative radii (would produce negative-volume cells)
            if np.any(line.r < 0):
                errors.append(f"{name}: negative radius detected")

            # Check segment lengths for extreme aspect ratios
            if len(line.r) > 1:
                dr = np.diff(line.r)
                dz = np.diff(line.z)
                seg_len = np.sqrt(dr ** 2 + dz ** 2)
                if seg_len.min() > 0:
                    aspect = seg_len.max() / seg_len.min()
                    if aspect > max_aspect_ratio:
                        warnings.append(
                            f"{name}: segment aspect ratio {aspect:.1f} "
                            f"exceeds limit {max_aspect_ratio}"
                        )

            # Check for sharp angles (excessive skew)
            if len(line.r) > 2:
                dx = np.column_stack([np.diff(line.r), np.diff(line.z)])
                for i in range(len(dx) - 1):
                    cos_a = np.dot(dx[i], dx[i + 1]) / (
                        np.linalg.norm(dx[i]) * np.linalg.norm(dx[i + 1]) + 1e-30
                    )
                    angle_deg = math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))
                    skew = abs(180.0 - angle_deg)
                    if skew > max_skew_deg:
                        warnings.append(
                            f"{name}: skew angle {skew:.1f}° at segment {i} "
                            f"exceeds limit {max_skew_deg}°"
                        )
                        break  # report only first occurrence per line

        return {"errors": errors, "warnings": warnings}

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _extend_inlet(
        self,
        rz: NDArray[np.float64],
        length: float,
        flaring_ratio: float,
    ) -> MeridionalLine:
        """Build the inlet extension section upstream from the leading edge."""
        r0, z0 = rz[0, 0], rz[0, 1]

        # Determine upstream direction from first two blade points
        if len(rz) > 1:
            dr = rz[0, 0] - rz[1, 0]
            dz = rz[0, 1] - rz[1, 1]
            mag = math.sqrt(dr ** 2 + dz ** 2)
            if mag > 1e-12:
                dr /= mag
                dz /= mag
            else:
                dr, dz = 0.0, -1.0  # default: axial upstream
        else:
            dr, dz = 0.0, -1.0

        n = self.n_extension_pts
        t = np.linspace(0.0, 1.0, n)
        base_r = r0 + dr * length * t
        base_z = z0 + dz * length * t

        # Apply flaring
        flare = _flaring_profile(n, flaring_ratio, self.flaring_type)
        r_ext = base_r * flare
        z_ext = base_z  # flaring only affects radial direction

        # Reverse so that the array goes from far-field → blade LE
        return MeridionalLine(r_ext[::-1].copy(), z_ext[::-1].copy())

    def _extend_outlet(
        self,
        rz: NDArray[np.float64],
        length: float,
        flaring_ratio: float,
    ) -> MeridionalLine:
        """Build the outlet extension section downstream from the trailing edge."""
        r_last, z_last = rz[-1, 0], rz[-1, 1]

        # Determine downstream direction from last two blade points
        if len(rz) > 1:
            dr = rz[-1, 0] - rz[-2, 0]
            dz = rz[-1, 1] - rz[-2, 1]
            mag = math.sqrt(dr ** 2 + dz ** 2)
            if mag > 1e-12:
                dr /= mag
                dz /= mag
            else:
                dr, dz = 0.0, 1.0
        else:
            dr, dz = 0.0, 1.0

        n = self.n_extension_pts
        t = np.linspace(0.0, 1.0, n)
        base_r = r_last + dr * length * t
        base_z = z_last + dz * length * t

        flare = _flaring_profile(n, flaring_ratio, self.flaring_type)
        r_ext = base_r * flare

        return MeridionalLine(r_ext.copy(), base_z.copy())

    @staticmethod
    def _resample_domain(domain: ExtendedDomain, factor: int) -> ExtendedDomain:
        """Resample every meridional line by the given factor."""

        def _resample_line(line: MeridionalLine, fac: int) -> MeridionalLine:
            n_orig = len(line.r)
            n_new = max(n_orig * fac, 3)
            t_old = np.linspace(0, 1, n_orig)
            t_new = np.linspace(0, 1, n_new)
            r_new = np.interp(t_new, t_old, line.r)
            z_new = np.interp(t_new, t_old, line.z)
            return MeridionalLine(r_new, z_new)

        return ExtendedDomain(
            inlet_hub=_resample_line(domain.inlet_hub, factor),
            inlet_shroud=_resample_line(domain.inlet_shroud, factor),
            blade_hub=_resample_line(domain.blade_hub, factor),
            blade_shroud=_resample_line(domain.blade_shroud, factor),
            outlet_hub=_resample_line(domain.outlet_hub, factor),
            outlet_shroud=_resample_line(domain.outlet_shroud, factor),
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PUMP_STANDARD = DomainExtent(
    inlet_extension=3.0,
    outlet_extension=5.0,
    inlet_hub_ratio=1.0,
    inlet_shroud_ratio=1.0,
    outlet_hub_ratio=1.0,
    outlet_shroud_ratio=1.0,
    flaring_type="linear",
)

PUMP_FINE = DomainExtent(
    inlet_extension=5.0,
    outlet_extension=8.0,
    inlet_hub_ratio=1.05,
    inlet_shroud_ratio=1.05,
    outlet_hub_ratio=1.05,
    outlet_shroud_ratio=1.05,
    flaring_type="parabolic",
)

TURBINE_STANDARD = DomainExtent(
    inlet_extension=4.0,
    outlet_extension=6.0,
    inlet_hub_ratio=1.1,
    inlet_shroud_ratio=1.1,
    outlet_hub_ratio=1.1,
    outlet_shroud_ratio=1.1,
    flaring_type="linear",
)

COMPRESSOR = DomainExtent(
    inlet_extension=3.0,
    outlet_extension=10.0,
    inlet_hub_ratio=1.15,
    inlet_shroud_ratio=1.15,
    outlet_hub_ratio=1.15,
    outlet_shroud_ratio=1.15,
    flaring_type="exponential",
)

PRESETS: dict[str, DomainExtent] = {
    "pump_standard": PUMP_STANDARD,
    "pump_fine": PUMP_FINE,
    "turbine_standard": TURBINE_STANDARD,
    "compressor": COMPRESSOR,
}
