"""Advanced volute geometries — double, rectangular, axial-entry, and shell.

Provides parametric volute variants beyond the standard single-circular:

- DoubleVolute: twin-entry 180-degree split, each passage handles half-flow.
- RectangularVolute: rectangular cross-sections for fan applications.
- AxialEntryVolute: axial inlet transitioning to circumferential scroll.
- VoluteShell: adds wall thickness to any volute geometry.

References:
    - Gulich (2014), Ch. 7 — Volute casing design.
    - Eck (1973) — Fans: rectangular volutes.
    - Stepanoff (1957) — Centrifugal and Axial Flow Pumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from hpe.geometry.volute.models import VoluteParams


# ---------------------------------------------------------------------------
# DoubleVolute
# ---------------------------------------------------------------------------

@dataclass
class DoubleVoluteConfig:
    """Configuration for a double (twin-entry) volute."""

    base: VoluteParams
    splitter_angle_deg: float = 180.0
    splitter_thickness: float = 0.005  # [m]
    splitter_gap_ratio: float = 0.02
    merge_length_ratio: float = 1.0
    merge_area_ratio: float = 1.1
    n_stations_per_passage: int = 18


@dataclass
class DoubleVolutePassage:
    """Area distribution for one passage of a double volute."""

    theta_deg: np.ndarray        # Angular stations [deg]
    areas: np.ndarray            # Cross-section area at each station [m^2]
    radii_outer: np.ndarray      # Outer radius at each station [m]
    widths: np.ndarray           # Section width at each station [m]
    hub_rz: np.ndarray           # Hub profile (r, z) pairs [m]
    shroud_rz: np.ndarray        # Shroud profile (r, z) pairs [m]


@dataclass
class DoubleVoluteResult:
    """Complete result for a double volute design."""

    passage_a: DoubleVolutePassage
    passage_b: DoubleVolutePassage
    splitter_rz: np.ndarray          # Splitter wall profile (r, z) [m]
    splitter_r_inner: float          # [m]
    splitter_r_outer: float          # [m]
    radial_force_ratio: float        # F_double / F_single
    merge_outlet_area: float         # [m^2]
    total_discharge_area: float      # [m^2]


class DoubleVolute:
    """Twin-entry volute with 180-degree split and inner splitter wall.

    Each passage handles half the flow. The splitter wall runs from
    the tongue to the 180-degree partition, reducing radial thrust
    at off-design conditions.
    """

    def __init__(self, config: DoubleVoluteConfig) -> None:
        self.config = config

    def generate(self) -> DoubleVoluteResult:
        """Generate both passages, splitter wall, and hub/shroud profiles.

        Returns:
            DoubleVoluteResult with complete geometry data.
        """
        cfg = self.config
        base = cfg.base
        r3 = base.r3
        r2 = base.d2 / 2.0
        K = r2 * base.cu2  # angular momentum constant
        Q_half = base.flow_rate / 2.0

        passage_a = self._size_passage(
            r3, K, Q_half, base,
            theta_start=0.0,
            theta_end=cfg.splitter_angle_deg,
        )
        passage_b = self._size_passage(
            r3, K, Q_half, base,
            theta_start=cfg.splitter_angle_deg,
            theta_end=360.0,
        )

        # Splitter wall profile: arc from r3 to max outer radius
        max_r = float(max(np.max(passage_a.radii_outer), np.max(passage_b.radii_outer)))
        splitter_r_inner = r3 + cfg.splitter_thickness
        splitter_r_outer = max_r * 0.95

        n_splitter = 20
        r_vals = np.linspace(splitter_r_inner, splitter_r_outer, n_splitter)
        z_vals = np.zeros(n_splitter)
        splitter_rz = np.column_stack([r_vals, z_vals])

        # Radial force ratio
        dev = abs(cfg.splitter_angle_deg - 180.0) / 180.0
        radial_force_ratio = 0.20 + 0.60 * dev ** 1.5
        radial_force_ratio = float(np.clip(radial_force_ratio, 0.15, 1.0))

        # Merge section
        area_a = float(passage_a.areas[-1])
        area_b = float(passage_b.areas[-1])
        merge_outlet = (area_a + area_b) * cfg.merge_area_ratio

        return DoubleVoluteResult(
            passage_a=passage_a,
            passage_b=passage_b,
            splitter_rz=splitter_rz,
            splitter_r_inner=splitter_r_inner,
            splitter_r_outer=splitter_r_outer,
            radial_force_ratio=radial_force_ratio,
            merge_outlet_area=merge_outlet,
            total_discharge_area=merge_outlet,
        )

    # ------------------------------------------------------------------

    def _size_passage(
        self,
        r3: float,
        K: float,
        Q: float,
        base: VoluteParams,
        theta_start: float,
        theta_end: float,
    ) -> DoubleVolutePassage:
        """Size a single passage by angular-momentum conservation."""
        n = self.config.n_stations_per_passage
        theta_range = theta_end - theta_start
        theta_deg = np.linspace(theta_start, theta_end, n + 1)
        theta_local_rad = np.radians(theta_deg - theta_start)
        total_rad = math.radians(theta_range)

        Q_theta = Q * theta_local_rad / total_rad
        V_mean = K / r3 if r3 > 0 else 1.0
        areas = Q_theta / V_mean

        # Tongue minimum area
        tongue_min = 0.001 * base.b2 * base.d2
        local_deg = theta_deg - theta_start
        min_areas = tongue_min * np.minimum(1.0, local_deg / 30.0)
        areas = np.maximum(areas, min_areas)

        # Section dimensions (circular)
        r_section = np.sqrt(areas / math.pi)
        widths = 2.0 * r_section
        radii_outer = r3 + 2.0 * r_section

        # Hub/shroud profiles: (r, z) along the passage centerline
        hub_rz = np.column_stack([radii_outer, -base.b2 / 2.0 * np.ones(len(theta_deg))])
        shroud_rz = np.column_stack([radii_outer, base.b2 / 2.0 * np.ones(len(theta_deg))])

        return DoubleVolutePassage(
            theta_deg=theta_deg,
            areas=areas,
            radii_outer=radii_outer,
            widths=widths,
            hub_rz=hub_rz,
            shroud_rz=shroud_rz,
        )


# ---------------------------------------------------------------------------
# RectangularVolute
# ---------------------------------------------------------------------------

@dataclass
class RectangularVoluteConfig:
    """Configuration for a rectangular cross-section volute (fan applications)."""

    base: VoluteParams
    aspect_ratio: float = 1.5             # height / width at discharge
    aspect_ratio_tongue: float = 2.0      # height / width near tongue
    corner_radius: float = 0.005          # Corner fillet radius [m]
    area_law: str = "linear"              # "linear" or "angular_momentum"
    n_stations: int = 36


@dataclass
class RectangularVoluteStation:
    """Cross-section data at one angular station."""

    theta_deg: float
    area: float       # [m^2]
    width: float      # [m]
    height: float     # [m]
    r_outer: float    # Outer radius [m]


@dataclass
class RectangularVoluteResult:
    """Complete result for a rectangular volute."""

    stations: List[RectangularVoluteStation]
    r3: float
    discharge_area: float
    discharge_width: float
    discharge_height: float
    corner_radius: float


class RectangularVolute:
    """Rectangular cross-section volute for fan applications.

    Width and height vary with circumferential angle. The aspect
    ratio transitions from ``aspect_ratio_tongue`` at the tongue
    to ``aspect_ratio`` at the discharge.
    """

    def __init__(self, config: RectangularVoluteConfig) -> None:
        self.config = config

    def generate(self) -> RectangularVoluteResult:
        """Generate rectangular volute station data.

        Returns:
            RectangularVoluteResult with station-by-station dimensions.
        """
        cfg = self.config
        base = cfg.base
        r3 = base.r3
        r2 = base.d2 / 2.0
        K = r2 * base.cu2
        Q = base.flow_rate

        stations: List[RectangularVoluteStation] = []

        for i in range(cfg.n_stations + 1):
            theta_deg = i * 360.0 / cfg.n_stations
            theta_rad = math.radians(theta_deg)
            frac = theta_deg / 360.0  # 0..1

            # Flow collected up to this angle
            Q_theta = Q * theta_rad / (2.0 * math.pi)

            # Area
            if cfg.area_law == "angular_momentum":
                V_mean = K / r3 if r3 > 0 else 1.0
                area = Q_theta / V_mean if V_mean > 0 else 0.0
            else:
                # Linear area growth
                V_mean = K / r3 if r3 > 0 else 1.0
                area_360 = Q / V_mean if V_mean > 0 else 0.0
                area = area_360 * frac

            # Minimum area near tongue
            tongue_min = 0.001 * base.b2 * base.d2
            area = max(area, tongue_min * min(1.0, theta_deg / 30.0))

            # Aspect ratio interpolation
            ar = cfg.aspect_ratio_tongue + (cfg.aspect_ratio - cfg.aspect_ratio_tongue) * frac

            # height = ar * width, area = width * height = ar * width^2
            width = math.sqrt(area / ar) if ar > 0 and area > 0 else 0.0
            height = ar * width

            r_outer = r3 + width

            stations.append(RectangularVoluteStation(
                theta_deg=theta_deg,
                area=area,
                width=width,
                height=height,
                r_outer=r_outer,
            ))

        discharge = stations[-1] if stations else stations[0]

        return RectangularVoluteResult(
            stations=stations,
            r3=r3,
            discharge_area=discharge.area,
            discharge_width=discharge.width,
            discharge_height=discharge.height,
            corner_radius=cfg.corner_radius,
        )


# ---------------------------------------------------------------------------
# AxialEntryVolute
# ---------------------------------------------------------------------------

@dataclass
class AxialEntryVoluteConfig:
    """Configuration for an axial-entry volute (compressor applications)."""

    base: VoluteParams
    axial_inlet_length: float = 0.0       # Auto-computed if 0 [m]
    axial_inlet_diameter: float = 0.0     # Auto-computed if 0 [m]
    blend_fraction: float = 0.3           # Fraction of scroll for transition region
    n_stations: int = 36


@dataclass
class AxialEntryVoluteResult:
    """Result for axial-entry volute."""

    # Scroll section (standard area distribution)
    theta_deg: np.ndarray
    areas: np.ndarray
    radii_outer: np.ndarray

    # Axial inlet
    axial_inlet_length: float     # [m]
    axial_inlet_diameter: float   # [m]
    axial_inlet_area: float       # [m^2]

    # Blend region
    blend_start_theta: float      # [deg]
    blend_end_theta: float        # [deg]
    blend_rz: np.ndarray          # Transition profile (r, z) [m]

    r3: float
    discharge_area: float


class AxialEntryVolute:
    """Axial-entry volute where flow enters axially and transitions
    to a circumferential scroll.

    Used in some compressor and blower applications where the inlet
    pipe is coaxial with the machine shaft.
    """

    def __init__(self, config: AxialEntryVoluteConfig) -> None:
        self.config = config

    def generate(self) -> AxialEntryVoluteResult:
        """Generate axial-entry volute geometry.

        Returns:
            AxialEntryVoluteResult with scroll, inlet, and blend data.
        """
        cfg = self.config
        base = cfg.base
        r3 = base.r3
        r2 = base.d2 / 2.0
        K = r2 * base.cu2
        Q = base.flow_rate

        # Auto-compute inlet dimensions if not specified
        inlet_dia = cfg.axial_inlet_diameter
        if inlet_dia <= 0:
            inlet_dia = base.d2 * 0.8

        inlet_area = math.pi * inlet_dia ** 2 / 4.0

        inlet_length = cfg.axial_inlet_length
        if inlet_length <= 0:
            inlet_length = inlet_dia * 1.5

        # Scroll area distribution
        n = cfg.n_stations
        theta_deg = np.linspace(0, 360, n + 1)
        theta_rad = np.radians(theta_deg)

        Q_theta = Q * theta_rad / (2.0 * math.pi)
        V_mean = K / r3 if r3 > 0 else 1.0
        areas = Q_theta / V_mean

        # Tongue minimum
        tongue_min = 0.001 * base.b2 * base.d2
        min_areas = tongue_min * np.minimum(1.0, theta_deg / 30.0)
        areas = np.maximum(areas, min_areas)

        r_section = np.sqrt(areas / math.pi)
        radii_outer = r3 + 2.0 * r_section

        # Blend region: transition from axial pipe to scroll
        blend_end_theta = 360.0 * cfg.blend_fraction
        n_blend = max(5, int(n * cfg.blend_fraction))
        blend_theta = np.linspace(0, blend_end_theta, n_blend)

        # Radial position transitions from inlet_dia/2 to r3
        blend_r = np.linspace(inlet_dia / 2.0, r3, n_blend)
        # Axial position transitions from -inlet_length to 0
        blend_z = np.linspace(-inlet_length, 0.0, n_blend)
        blend_rz = np.column_stack([blend_r, blend_z])

        return AxialEntryVoluteResult(
            theta_deg=theta_deg,
            areas=areas,
            radii_outer=radii_outer,
            axial_inlet_length=inlet_length,
            axial_inlet_diameter=inlet_dia,
            axial_inlet_area=inlet_area,
            blend_start_theta=0.0,
            blend_end_theta=blend_end_theta,
            blend_rz=blend_rz,
            r3=r3,
            discharge_area=float(areas[-1]),
        )


# ---------------------------------------------------------------------------
# VoluteShell
# ---------------------------------------------------------------------------

@dataclass
class VoluteShellConfig:
    """Configuration for adding wall thickness to a volute."""

    thickness_uniform: float = 0.008          # Uniform wall thickness [m]
    thickness_tongue: Optional[float] = None  # Thickness at tongue (thicker for wear) [m]
    thickness_discharge: Optional[float] = None  # Thickness at discharge [m]
    n_circumferential: int = 36
    n_profile: int = 20                        # Points per cross-section profile


@dataclass
class VoluteShellResult:
    """Inner and outer surface meshes for structural/casting analysis."""

    # Each is shape (n_circ, n_profile, 3) in (x, y, z)
    inner_surface: np.ndarray
    outer_surface: np.ndarray
    thickness_distribution: np.ndarray   # Thickness at each station [m]
    theta_stations: np.ndarray           # [deg]


class VoluteShell:
    """Adds wall thickness to any volute geometry for structural analysis.

    Given a volute area distribution (inner flow surface), offsets
    outward by the configured thickness to produce the outer casting
    surface. Thickness can be uniform or varying (thicker at tongue).
    """

    def __init__(self, config: VoluteShellConfig) -> None:
        self.config = config

    def generate(
        self,
        r3: float,
        areas: np.ndarray,
        b2: float,
        theta_stations_deg: Optional[np.ndarray] = None,
    ) -> VoluteShellResult:
        """Generate inner and outer surface meshes.

        Args:
            r3: Base (inlet) radius of volute [m].
            areas: Cross-section area at each angular station [m^2].
            b2: Impeller outlet width [m].
            theta_stations_deg: Angular stations [deg]. If None, evenly spaced.

        Returns:
            VoluteShellResult with surface meshes and thickness data.
        """
        cfg = self.config
        n_circ = len(areas)

        if theta_stations_deg is None:
            theta_stations_deg = np.linspace(0, 360, n_circ)

        # Thickness distribution
        t_uniform = cfg.thickness_uniform
        t_tongue = cfg.thickness_tongue if cfg.thickness_tongue is not None else t_uniform * 1.5
        t_discharge = cfg.thickness_discharge if cfg.thickness_discharge is not None else t_uniform

        # Interpolate: tongue (theta=0) -> uniform (mid) -> discharge (theta=360)
        frac = theta_stations_deg / 360.0
        # Piecewise: 0->0.1 tongue region, 0.1->0.9 uniform, 0.9->1.0 discharge
        thickness = np.full(n_circ, t_uniform)
        tongue_mask = frac < 0.1
        discharge_mask = frac > 0.9
        thickness[tongue_mask] = t_tongue + (t_uniform - t_tongue) * (frac[tongue_mask] / 0.1)
        thickness[discharge_mask] = t_uniform + (t_discharge - t_uniform) * ((frac[discharge_mask] - 0.9) / 0.1)

        n_prof = cfg.n_profile
        inner_surface = np.zeros((n_circ, n_prof, 3))
        outer_surface = np.zeros((n_circ, n_prof, 3))

        for i in range(n_circ):
            theta_rad = math.radians(float(theta_stations_deg[i]))
            area = float(areas[i])
            t = float(thickness[i])

            # Inner section radius (circular cross-section)
            r_inner = math.sqrt(area / math.pi) if area > 0 else 0.001
            r_outer = r_inner + t

            # Center of section on the spiral
            r_center = r3 + r_inner
            cx = r_center * math.cos(theta_rad)
            cy = r_center * math.sin(theta_rad)

            # Profile points around the section
            profile_angles = np.linspace(0, 2 * math.pi, n_prof, endpoint=False)

            for j in range(n_prof):
                pa = float(profile_angles[j])
                # Local offsets (in radial-axial plane)
                dr_inner = r_inner * math.cos(pa)
                dz_inner = r_inner * math.sin(pa)
                dr_outer = r_outer * math.cos(pa)
                dz_outer = r_outer * math.sin(pa)

                # Inner surface point
                r_pt_inner = r_center + dr_inner
                inner_surface[i, j, 0] = r_pt_inner * math.cos(theta_rad)
                inner_surface[i, j, 1] = r_pt_inner * math.sin(theta_rad)
                inner_surface[i, j, 2] = dz_inner

                # Outer surface point
                r_pt_outer = r_center + dr_outer
                outer_surface[i, j, 0] = r_pt_outer * math.cos(theta_rad)
                outer_surface[i, j, 1] = r_pt_outer * math.sin(theta_rad)
                outer_surface[i, j, 2] = dz_outer

        return VoluteShellResult(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            thickness_distribution=thickness,
            theta_stations=theta_stations_deg,
        )
