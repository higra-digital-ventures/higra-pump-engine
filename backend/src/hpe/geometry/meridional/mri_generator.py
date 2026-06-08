"""Parametric Meridional (MRI) Generator — hub/shroud curve definition.

Generates meridional profiles for turbomachinery rotors using
parametric control points with Bezier curves and line segments.

The meridional plane is defined in (r, z) coordinates where:
    r = radial distance from axis of rotation [m]
    z = axial distance (positive = downstream) [m]

Templates are provided for common machine types:
    RADIAL_PUMP, MIXED_FLOW, FRANCIS_TURBINE, AXIAL.

References:
    - Gulich (2014), Ch. 3 — Impeller design.
    - Pfleiderer & Petermann (2005) — Stroemungsmaschinen.
    - Aungier (2000) — Centrifugal Compressors, Ch. 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MeridionalProfile:
    """Hub and shroud curves in the meridional (r, z) plane."""

    hub_rz: np.ndarray      # shape (n, 2) — (r, z) pairs [m]
    shroud_rz: np.ndarray   # shape (n, 2) — (r, z) pairs [m]
    n_points: int


@dataclass
class MRIParams:
    """Control parameters for meridional profile generation."""

    inlet_radius: float          # r at inlet (shroud tip) [m]
    outlet_radius: float         # r at outlet (impeller exit) [m]
    axial_length: float          # Total axial extent [m]
    hub_curvature: float = 0.5   # 0 = sharp corner, 1 = full circular arc
    shroud_curvature: float = 0.5
    inlet_angle_hub: float = 90.0    # Hub angle at inlet [deg] (90 = radial)
    inlet_angle_shroud: float = 90.0 # Shroud angle at inlet [deg]
    outlet_angle_hub: float = 90.0   # Hub angle at outlet [deg] (90 = radial)
    outlet_angle_shroud: float = 90.0
    hub_inlet_radius: float = 0.0    # Hub r at inlet (shaft radius). 0 = auto
    passage_width_inlet: float = 0.0 # b1 at inlet [m]. 0 = auto from inlet_radius


@dataclass
class MRIValidationResult:
    """Result of meridional geometry validation checks."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    min_passage_width: float       # Minimum hub-to-shroud distance [m]
    max_curvature_hub: float       # Maximum curvature of hub [1/m]
    max_curvature_shroud: float    # Maximum curvature of shroud [1/m]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

class MRITemplate(str, Enum):
    """Preset meridional profile templates."""

    RADIAL_PUMP = "radial_pump"
    MIXED_FLOW = "mixed_flow"
    FRANCIS_TURBINE = "francis_turbine"
    AXIAL = "axial"

    def to_params(self, d2: float = 0.3, b2: float = 0.02) -> MRIParams:
        """Convert template to MRIParams using reference dimensions.

        Args:
            d2: Impeller outlet diameter [m] (reference scale).
            b2: Impeller outlet width [m].

        Returns:
            MRIParams configured for this template.
        """
        r2 = d2 / 2.0

        if self == MRITemplate.RADIAL_PUMP:
            return MRIParams(
                inlet_radius=r2 * 0.45,
                outlet_radius=r2,
                axial_length=d2 * 0.3,
                hub_curvature=0.6,
                shroud_curvature=0.7,
                inlet_angle_hub=90.0,
                inlet_angle_shroud=90.0,
                outlet_angle_hub=90.0,
                outlet_angle_shroud=90.0,
                passage_width_inlet=b2 * 1.2,
            )
        elif self == MRITemplate.MIXED_FLOW:
            return MRIParams(
                inlet_radius=r2 * 0.50,
                outlet_radius=r2,
                axial_length=d2 * 0.5,
                hub_curvature=0.5,
                shroud_curvature=0.6,
                inlet_angle_hub=70.0,
                inlet_angle_shroud=75.0,
                outlet_angle_hub=45.0,
                outlet_angle_shroud=50.0,
                passage_width_inlet=b2 * 1.4,
            )
        elif self == MRITemplate.FRANCIS_TURBINE:
            return MRIParams(
                inlet_radius=r2 * 0.90,
                outlet_radius=r2,
                axial_length=d2 * 0.7,
                hub_curvature=0.8,
                shroud_curvature=0.8,
                inlet_angle_hub=80.0,
                inlet_angle_shroud=85.0,
                outlet_angle_hub=90.0,
                outlet_angle_shroud=90.0,
                passage_width_inlet=b2 * 2.0,
            )
        else:  # AXIAL
            return MRIParams(
                inlet_radius=r2 * 0.85,
                outlet_radius=r2 * 0.85,
                axial_length=d2 * 0.25,
                hub_curvature=0.0,
                shroud_curvature=0.0,
                inlet_angle_hub=0.0,
                inlet_angle_shroud=0.0,
                outlet_angle_hub=0.0,
                outlet_angle_shroud=0.0,
                passage_width_inlet=b2 * 1.0,
            )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MRIGenerator:
    """Parametric meridional profile generator.

    Constructs hub and shroud curves using a combination of line
    segments and cubic Bezier curves. The curvature parameter
    controls the blending between a sharp-corner path and a smooth
    arc.

    Usage::

        gen = MRIGenerator(params)
        profile = gen.generate(n_points=50)
        validation = gen.validate()
    """

    def __init__(self, params: MRIParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_points: int = 50) -> MeridionalProfile:
        """Generate hub and shroud meridional curves.

        Args:
            n_points: Number of points along each curve.

        Returns:
            MeridionalProfile with hub_rz and shroud_rz arrays.
        """
        p = self.params

        # Derive hub inlet radius
        hub_r_inlet = p.hub_inlet_radius
        if hub_r_inlet <= 0:
            hub_r_inlet = p.inlet_radius * 0.3  # Default shaft radius

        # Derive passage width at inlet
        b1 = p.passage_width_inlet
        if b1 <= 0:
            b1 = p.inlet_radius - hub_r_inlet

        shroud_r_inlet = hub_r_inlet + b1

        # Hub control points in (r, z)
        # Inlet: (hub_r_inlet, 0), Outlet: (outlet_radius, axial_length)
        hub_rz = self._generate_curve(
            r_start=hub_r_inlet,
            z_start=0.0,
            r_end=p.outlet_radius * 0.85,  # Hub is inner wall
            z_end=p.axial_length,
            curvature=p.hub_curvature,
            angle_start=p.inlet_angle_hub,
            angle_end=p.outlet_angle_hub,
            n_points=n_points,
        )

        # Shroud control points
        shroud_rz = self._generate_curve(
            r_start=shroud_r_inlet,
            z_start=0.0,
            r_end=p.outlet_radius,
            z_end=p.axial_length,
            curvature=p.shroud_curvature,
            angle_start=p.inlet_angle_shroud,
            angle_end=p.outlet_angle_shroud,
            n_points=n_points,
        )

        return MeridionalProfile(
            hub_rz=hub_rz,
            shroud_rz=shroud_rz,
            n_points=n_points,
        )

    @classmethod
    def from_sizing_result(cls, sizing: object) -> MRIGenerator:
        """Auto-generate meridional parameters from a SizingResult.

        Maps D1, D2, b1, b2 from sizing to MRIParams.

        Args:
            sizing: A SizingResult object with impeller dimensions.

        Returns:
            MRIGenerator configured from the sizing data.
        """
        d1 = getattr(sizing, "impeller_d1", 0.0)
        d2 = getattr(sizing, "impeller_d2", 0.3)
        b1 = getattr(sizing, "impeller_b1", 0.0)
        b2 = getattr(sizing, "impeller_b2", 0.02)
        shaft_d = getattr(sizing, "shaft_diameter", 0.0)

        r1 = d1 / 2.0 if d1 > 0 else d2 * 0.22
        r2 = d2 / 2.0
        if b1 <= 0:
            b1 = b2 * 1.3

        hub_r_inlet = shaft_d / 2.0 if shaft_d > 0 else r1 * 0.4
        axial_length = d2 * 0.3  # Typical L/D2 for radial pump

        params = MRIParams(
            inlet_radius=r1,
            outlet_radius=r2,
            axial_length=axial_length,
            hub_curvature=0.6,
            shroud_curvature=0.7,
            inlet_angle_hub=90.0,
            inlet_angle_shroud=90.0,
            outlet_angle_hub=90.0,
            outlet_angle_shroud=90.0,
            hub_inlet_radius=hub_r_inlet,
            passage_width_inlet=b1,
        )
        return cls(params)

    def scale(self, factor: float) -> MeridionalProfile:
        """Generate and uniformly scale the meridional profile.

        Args:
            factor: Scale factor (e.g. 2.0 = double size).

        Returns:
            Scaled MeridionalProfile.
        """
        profile = self.generate()
        profile.hub_rz = profile.hub_rz * factor
        profile.shroud_rz = profile.shroud_rz * factor
        return profile

    def translate(self, dr: float, dz: float) -> MeridionalProfile:
        """Generate and translate the meridional profile.

        Args:
            dr: Radial offset [m].
            dz: Axial offset [m].

        Returns:
            Translated MeridionalProfile.
        """
        profile = self.generate()
        offset = np.array([dr, dz])
        profile.hub_rz = profile.hub_rz + offset
        profile.shroud_rz = profile.shroud_rz + offset
        return profile

    def validate(self, n_points: int = 50) -> MRIValidationResult:
        """Validate the meridional geometry.

        Checks for:
            - Hub/shroud intersection (negative passage width)
            - Minimum passage width too small
            - Excessive curvature

        Args:
            n_points: Resolution for validation.

        Returns:
            MRIValidationResult with diagnostics.
        """
        profile = self.generate(n_points=n_points)
        errors: List[str] = []
        warnings: List[str] = []

        hub = profile.hub_rz
        shroud = profile.shroud_rz

        # Passage width: radial distance between hub and shroud at each z
        passage_widths = shroud[:, 0] - hub[:, 0]
        min_pw = float(np.min(passage_widths))

        if min_pw < 0:
            errors.append(
                f"Hub and shroud intersect: minimum passage width = {min_pw:.4f} m"
            )
        elif min_pw < 0.001:
            warnings.append(
                f"Very narrow passage: minimum width = {min_pw:.4f} m"
            )

        # Curvature of hub and shroud
        kappa_hub = self._compute_max_curvature(hub)
        kappa_shroud = self._compute_max_curvature(shroud)

        r2 = self.params.outlet_radius
        if r2 > 0:
            if kappa_hub > 20.0 / r2:
                warnings.append(
                    f"High hub curvature: {kappa_hub:.1f} 1/m (limit ~{20.0/r2:.1f})"
                )
            if kappa_shroud > 20.0 / r2:
                warnings.append(
                    f"High shroud curvature: {kappa_shroud:.1f} 1/m (limit ~{20.0/r2:.1f})"
                )

        valid = len(errors) == 0

        return MRIValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            min_passage_width=min_pw,
            max_curvature_hub=kappa_hub,
            max_curvature_shroud=kappa_shroud,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_curve(
        r_start: float,
        z_start: float,
        r_end: float,
        z_end: float,
        curvature: float,
        angle_start: float,
        angle_end: float,
        n_points: int,
    ) -> np.ndarray:
        """Generate a meridional curve using cubic Bezier interpolation.

        The curvature parameter blends between a straight line (0)
        and a smooth arc with control points determined by the
        inlet/outlet angles.

        Args:
            r_start, z_start: Start point in (r, z).
            r_end, z_end: End point in (r, z).
            curvature: Blend factor 0..1 (0=straight, 1=full Bezier arc).
            angle_start: Tangent angle at start [deg] (90=radial, 0=axial).
            angle_end: Tangent angle at end [deg].
            n_points: Number of output points.

        Returns:
            ndarray of shape (n_points, 2) with (r, z) values.
        """
        # Straight-line distance for control-point scaling
        dist = math.sqrt((r_end - r_start) ** 2 + (z_end - z_start) ** 2)
        arm = dist * 0.4 * max(0.0, min(1.0, curvature))

        # Convert angles to tangent vectors (angle from +z axis)
        a0 = math.radians(angle_start)
        a1 = math.radians(angle_end)

        # Control point 1: offset from start along start tangent
        cp1_r = r_start + arm * math.sin(a0)
        cp1_z = z_start + arm * math.cos(a0)

        # Control point 2: offset from end opposite to end tangent
        cp2_r = r_end - arm * math.sin(a1)
        cp2_z = z_end - arm * math.cos(a1)

        # Cubic Bezier evaluation
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        p0 = np.array([[r_start, z_start]])
        p1 = np.array([[cp1_r, cp1_z]])
        p2 = np.array([[cp2_r, cp2_z]])
        p3 = np.array([[r_end, z_end]])

        curve = (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )

        return curve

    @staticmethod
    def _compute_max_curvature(pts: np.ndarray) -> float:
        """Compute maximum curvature of a 2D curve.

        Uses finite differences: kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2).

        Args:
            pts: Array of shape (n, 2).

        Returns:
            Maximum curvature [1/m].
        """
        if len(pts) < 3:
            return 0.0

        dr = np.gradient(pts[:, 0])
        dz = np.gradient(pts[:, 1])
        ddr = np.gradient(dr)
        ddz = np.gradient(dz)

        numerator = np.abs(dr * ddz - dz * ddr)
        denominator = (dr ** 2 + dz ** 2) ** 1.5

        # Avoid division by zero
        valid = denominator > 1e-12
        if not np.any(valid):
            return 0.0

        kappa = np.zeros_like(numerator)
        kappa[valid] = numerator[valid] / denominator[valid]

        return float(np.max(kappa))
