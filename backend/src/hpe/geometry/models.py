"""Data models for parametric geometry generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MeridionalChannel:
    """Meridional channel defined by hub and shroud curves in the (r, z) plane.

    Points are ordered from inlet (axial entry) to outlet (radial exit).
    r = radial coordinate [m], z = axial coordinate [m].
    Convention: z=0 at outlet plane, z positive toward inlet (upstream).
    """

    hub_points: list[tuple[float, float]]  # (r, z) from inlet to outlet
    shroud_points: list[tuple[float, float]]  # (r, z) from inlet to outlet


@dataclass
class BladeProfile:
    """2D blade profile in the (r, theta) plane.

    Represents the camber line and thickness distribution of one blade.
    """

    camber_points: list[tuple[float, float]]  # (r, theta) camber line
    pressure_side: list[tuple[float, float]]  # (r, theta) pressure side
    suction_side: list[tuple[float, float]]  # (r, theta) suction side
    thickness: float  # Maximum thickness [m]


@dataclass
class RunnerGeometryParams:
    """Consolidated input parameters for runner geometry generation.

    Can be constructed from a SizingResult or defined manually.
    """

    # Diameters [m]
    d2: float  # Outlet diameter
    d1: float  # Inlet (eye) diameter
    d1_hub: float  # Hub diameter at inlet

    # Widths [m]
    b2: float  # Outlet width
    b1: float  # Inlet width

    # Blade parameters
    beta1: float  # Inlet blade angle [deg]
    beta2: float  # Outlet blade angle [deg]
    blade_count: int  # Number of blades
    blade_thickness: float = 0.003  # Blade thickness [m] (default 3mm)

    # Hub/shroud
    hub_fillet_radius: float = 0.005  # Fillet radius at hub [m]
    axial_length: float | None = None  # If None, auto-calculated

    @classmethod
    def from_sizing_result(cls, sizing_result: object) -> RunnerGeometryParams:
        """Create RunnerGeometryParams from a SizingResult.

        Args:
            sizing_result: SizingResult from the sizing module.

        Returns:
            RunnerGeometryParams ready for geometry generation.
        """
        sr = sizing_result
        mp = sr.meridional_profile  # type: ignore[attr-defined]

        # Auto-calculate blade thickness as ~2% of D2
        d2 = sr.impeller_d2  # type: ignore[attr-defined]
        thickness = max(0.002, d2 * 0.02)

        return cls(
            d2=d2,
            d1=sr.impeller_d1,  # type: ignore[attr-defined]
            d1_hub=mp.get("d1_hub", sr.impeller_d1 * 0.35),  # type: ignore[attr-defined]
            b2=sr.impeller_b2,  # type: ignore[attr-defined]
            b1=mp.get("b1", sr.impeller_b2 * 1.2),  # type: ignore[attr-defined]
            beta1=sr.beta1,  # type: ignore[attr-defined]
            beta2=sr.beta2,  # type: ignore[attr-defined]
            blade_count=sr.blade_count,  # type: ignore[attr-defined]
            blade_thickness=thickness,
        )
