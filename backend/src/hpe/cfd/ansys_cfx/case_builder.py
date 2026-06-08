"""ANSYS CFX case builder — generates CCL, DEF setup scripts, and TurboGrid params.

Produces ready-to-import files for ANSYS CFX-Pre and TurboGrid from
HPE sizing results. Users run ANSYS externally and import results back.

References:
    ANSYS CFX-Pre User's Guide — CCL syntax.
    ANSYS TurboGrid User's Guide — machine definition.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Any

from hpe.core.models import SizingResult


# ---------------------------------------------------------------------------
# Domain parameter defaults
# ---------------------------------------------------------------------------

@dataclass
class CFXDomainParams:
    """Parameters controlling the CFX domain and solver setup."""

    fluid_density: float = 998.2       # kg/m3
    fluid_viscosity: float = 1.003e-3  # Pa.s (dynamic)
    fluid_name: str = "Water"
    reference_pressure: float = 101325.0  # Pa (1 atm)
    outlet_pressure: float = 0.0       # Gauge pressure at outlet [Pa]
    turbulence_model: str = "SST"      # SST | k-Epsilon
    advection_scheme: str = "High Resolution"
    timescale_control: str = "Auto Timescale"
    max_iterations: int = 500
    residual_target: float = 1.0e-5
    n_blade_passages: int = 1          # Periodic single-passage by default


class CFXCaseBuilder:
    """Generates ANSYS CFX configuration files from HPE sizing data."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_ccl(
        self,
        sizing_result: SizingResult,
        rpm: float,
        flow_rate: float,
        fluid_density: float = 998.0,
        domain_params: CFXDomainParams | None = None,
        single_passage: bool = True,
        roughness_um: float = 0.0,
    ) -> str:
        """Generate ANSYS CFX Command Language (.ccl) file content.

        The CCL defines the complete simulation setup: domain, boundary
        conditions, solver control, and output control.  It can be
        imported directly into CFX-Pre via *File > Import > CCL*.

        Args:
            sizing_result: Output from hpe.sizing.
            rpm: Rotational speed [rev/min].
            flow_rate: Volumetric flow rate [m3/s].
            fluid_density: Fluid density [kg/m3].
            domain_params: Optional solver/domain overrides.
            single_passage: If True (default), generate periodic single-passage
                domain with MRF. If False, generate full 360-degree domain.
            roughness_um: Wall roughness in micrometers. 0 = smooth wall.

        Returns:
            CCL file content as a string.
        """
        # --- Input validation ---
        if rpm <= 0 or rpm > 50000:
            raise ValueError(f"RPM {rpm} outside valid range 1-50000")
        if flow_rate <= 0:
            raise ValueError("Flow rate must be positive")

        dp = domain_params or CFXDomainParams()
        dp.fluid_density = fluid_density

        omega_rad = rpm * math.pi / 30.0
        mass_flow = fluid_density * flow_rate

        mp = sizing_result.meridional_profile
        d_inlet = sizing_result.impeller_d1
        d_hub = mp.get("d1_hub", d_inlet * 0.35)

        # Velocity components for inlet BC (cylindrical)
        vt_in = sizing_result.velocity_triangles.get("inlet", {})
        cm1 = vt_in.get("cm", flow_rate / (math.pi / 4 * (d_inlet**2 - d_hub**2)))
        cu1 = vt_in.get("cu", 0.0)  # pre-swirl (0 for no pre-swirl)

        z_blades = sizing_result.blade_count
        pitch_deg = 360.0 / z_blades

        # Wall roughness block
        roughness_block = self._build_roughness_block(roughness_um)

        # Build domain-specific sections
        if single_passage:
            domain_motion_option = "Rotating"
            periodic_section = self._build_periodic_section()
            analysis_type = "Steady State"
        else:
            domain_motion_option = "Rotating"
            periodic_section = ""  # No periodic BCs for full wheel
            analysis_type = "Steady State"

        # Inlet BC with cylindrical velocity components (swirl support)
        inlet_bc = textwrap.dedent(f"""\
            BOUNDARY: Inlet
              Boundary Type = INLET
              Location = inflow
              BOUNDARY CONDITIONS:
                FLOW REGIME:
                  Option = Subsonic
                END
                MASS AND MOMENTUM:
                  Option = Cylindrical Velocity Components
                  Axial Component = {cm1:.6f} [m s^-1]
                  Radial Component = 0.0 [m s^-1]
                  Theta Component = {cu1:.6f} [m s^-1]
                END
                TURBULENCE:
                  Option = Medium Intensity and Eddy Viscosity Ratio
                END
              END
            END""")

        # Outlet BC with radial equilibrium
        outlet_bc = textwrap.dedent(f"""\
            BOUNDARY: Outlet
              Boundary Type = OUTLET
              Location = outflow
              BOUNDARY CONDITIONS:
                FLOW REGIME:
                  Option = Subsonic
                END
                MASS AND MOMENTUM:
                  Option = Static Pressure
                  Relative Pressure = {dp.outlet_pressure:.1f} [Pa]
                END
              END
            END""")

        # Wall BCs with roughness option
        blade_bc = textwrap.dedent(f"""\
            BOUNDARY: Blade
              Boundary Type = WALL
              Location = blade
              BOUNDARY CONDITIONS:
                WALL INFLUENCE ON FLOW:
                  Option = No Slip Wall
                END
{roughness_block}
              END
            END""")

        hub_bc = textwrap.dedent(f"""\
            BOUNDARY: Hub
              Boundary Type = WALL
              Location = hub
              BOUNDARY CONDITIONS:
                WALL INFLUENCE ON FLOW:
                  Option = No Slip Wall
                END
{roughness_block}
              END
            END""")

        shroud_bc = textwrap.dedent(f"""\
            BOUNDARY: Shroud
              Boundary Type = WALL
              Location = shroud
              BOUNDARY CONDITIONS:
                MASS AND MOMENTUM:
                  Option = No Slip Wall
                END
                WALL INFLUENCE ON FLOW:
                  Option = No Slip Wall
                END
                WALL VELOCITY:
                  Option = Counter Rotating Wall
                END
{roughness_block}
              END
            END""")

        # Head expression for monitor
        head_expr = (
            "(massFlowAve(Total Pressure in Stn Frame)@Outlet "
            "- massFlowAve(Total Pressure in Stn Frame)@Inlet) "
            f"/ ({dp.fluid_density:.1f} * 9.80665)"
        )

        ccl = textwrap.dedent(f"""\
            # =========================================================
            # ANSYS CFX Command Language (CCL) — generated by HPE
            #
            # NOTE: Boundary "Location" names below follow TurboGrid
            # default conventions (lowercase). If your mesh uses
            # different region names, update the Location fields
            # to match (e.g. inflow, outflow, blade, hub, shroud,
            # periodic1side1, periodic1side2).
            # =========================================================
            #
            # Pump: centrifugal, Nq = {sizing_result.specific_speed_nq:.1f}
            # D2 = {sizing_result.impeller_d2 * 1000:.1f} mm,
            # RPM = {rpm:.0f}
            # Mode: {"single-passage (periodic)" if single_passage else "full-wheel (360 deg)"}
            # Roughness: {roughness_um:.1f} um {"(smooth)" if roughness_um <= 0 else "(sand-grain)"}
            # =========================================================

            LIBRARY:
              MATERIAL: {dp.fluid_name}
                Material Description = {dp.fluid_name} at 25 degC
                Material Group = Fluid
                Option = Pure Substance
                Thermodynamic State = Liquid
                PROPERTIES:
                  Option = General Material
                  EQUATION OF STATE:
                    Density = {dp.fluid_density:.1f} [kg m^-3]
                    Molar Mass = 18.02 [kg kmol^-1]
                    Option = Value
                  END
                  DYNAMIC VISCOSITY:
                    Dynamic Viscosity = {dp.fluid_viscosity:.6e} [Pa s]
                    Option = Value
                  END
                END
              END
            END

            FLOW: Flow Analysis 1
              SOLUTION UNITS:
                Angle Units = [rad]
                Length Units = [m]
                Mass Units = [kg]
                Solid Angle Units = [sr]
                Temperature Units = [K]
                Time Units = [s]
              END

              Analysis Type = {analysis_type}

              &replace DOMAIN: Impeller
                Coord Frame = Coord 0
                Domain Type = Fluid
                Fluids List = {dp.fluid_name}
                Location = Assembly

                DOMAIN MODELS:
                  BUOYANCY MODEL:
                    Option = Non Buoyant
                  END
                  DOMAIN MOTION:
                    Angular Velocity = {omega_rad:.6f} [radian s^-1]
                    Axis Definition = Coordinate Axis
                    Option = {domain_motion_option}
                    AXIS DEFINITION:
                      Option = Coordinate Axis
                      Rotation Axis = Z
                    END
                  END
                  MESH DEFORMATION:
                    Option = None
                  END
                  REFERENCE PRESSURE:
                    Reference Pressure = {dp.reference_pressure:.1f} [Pa]
                  END
                END

                FLUID MODELS:
                  HEAT TRANSFER MODEL:
                    Option = None
                  END
                  TURBULENCE MODEL:
                    Option = {dp.turbulence_model}
                  END
                  TURBULENT WALL FUNCTIONS:
                    Option = Automatic
                  END
                END

                # --- Boundary Conditions ---

        """)

        # Indent BCs to match domain nesting
        bc_indent = "        "
        for bc_text in [inlet_bc, outlet_bc, blade_bc, hub_bc, shroud_bc]:
            for line in bc_text.splitlines():
                ccl += bc_indent + line + "\n"
            ccl += "\n"

        # Add periodic section only for single-passage
        if single_passage:
            for line in periodic_section.splitlines():
                ccl += bc_indent + line + "\n"
            ccl += "\n"

        # Close DOMAIN
        ccl += "      END  # DOMAIN\n\n"

        # Solver control
        ccl += textwrap.dedent(f"""\
              # --- Solver Control ---

              SOLVER CONTROL:
                Turbulence Numerics = First Order
                ADVECTION SCHEME:
                  Option = {dp.advection_scheme}
                END
                CONVERGENCE CONTROL:
                  Maximum Number of Iterations = {dp.max_iterations}
                  Timescale Control = {dp.timescale_control}
                END
                CONVERGENCE CRITERIA:
                  Residual Target = {dp.residual_target:.1e}
                  Residual Type = RMS
                END
                DYNAMIC MODEL CONTROL:
                  Global Dynamic Model Control = On
                END
              END

              # --- Output Control ---

              OUTPUT CONTROL:
                RESULTS:
                  Option = Standard
                  File Compression Level = Default
                  OUTPUT BOUNDARY:
                    Option = All
                    FIELD DATA:
                      Output Var List = Pressure, Velocity, Turbulence Kinetic Energy
                    END
                  END
                END

                MONITOR OBJECTS:
                  MONITOR BALANCES:
                    Option = Full
                  END
                  MONITOR FORCES:
                    Option = Full
                  END
                  MONITOR TOTALS:
                    Option = Full
                  END
                  MONITOR POINT: Inlet Pressure
                    Coord Frame = Coord 0
                    Expression Value = massFlowAve(Total Pressure in Stn Frame)@Inlet
                    Option = Expression
                  END
                  MONITOR POINT: Outlet Pressure
                    Coord Frame = Coord 0
                    Expression Value = massFlowAve(Total Pressure in Stn Frame)@Outlet
                    Option = Expression
                  END
                  MONITOR POINT: Torque
                    Coord Frame = Coord 0
                    Expression Value = torque_z()@Blade
                    Option = Expression
                  END
                  MONITOR POINT: Efficiency
                    Coord Frame = Coord 0
                    Expression Value = \\
                      (massFlowAve(Total Pressure in Stn Frame)@Outlet \\
                       - massFlowAve(Total Pressure in Stn Frame)@Inlet) \\
                      * massFlow()@Inlet / {dp.fluid_density:.1f} \\
                      / (torque_z()@Blade * {omega_rad:.6f})
                    Option = Expression
                  END
                  MONITOR POINT: Head
                    Coord Frame = Coord 0
                    Expression Value = {head_expr}
                    Option = Expression
                  END
                END
              END

            END  # FLOW
        """)
        return ccl

    def generate_post_template(
        self,
        sizing_result: SizingResult,
        rpm: float,
        flow_rate: float,
        fluid_density: float = 998.0,
    ) -> str:
        """Generate CFX-Post session file (.cse) for standard results.

        Produces a post-processing template that creates standard
        turbomachinery visualizations and exports key metrics.

        Args:
            sizing_result: Output from hpe.sizing.
            rpm: Rotational speed [rev/min].
            flow_rate: Volumetric flow rate [m3/s].
            fluid_density: Fluid density [kg/m3].

        Returns:
            CFX-Post session (.cse) content as a string.
        """
        omega = rpm * 2.0 * math.pi / 60.0
        head = sizing_result.velocity_triangles.get("outlet", {}).get("head", 0.0)
        z_blades = sizing_result.blade_count

        template = textwrap.dedent(f"""\
            # =========================================================
            # CFX-Post Session Template — HPE Generated
            # Standard turbomachinery post-processing
            # Pump: Nq = {sizing_result.specific_speed_nq:.1f}
            # RPM = {rpm:.0f}, Q = {flow_rate * 3600:.1f} m3/h
            # =========================================================

            # Create blade loading chart
            CHART: Blade Loading
              Data Source = Default Domain
              X Axis Expression = Normalized Streamwise
              Y Axis Expression = Pressure
              Boundary = Blade
            END

            # Create meridional velocity contour
            CONTOUR: Meridional Velocity
              Domain = Default Domain
              Location = Hub to Shroud (mid-span)
              Variable = Velocity in Stn Frame
            END

            # Create pressure contour on blade surfaces
            CONTOUR: Blade Pressure
              Domain = Default Domain
              Location = Blade
              Variable = Pressure
            END

            # Create turbulence kinetic energy contour
            CONTOUR: TKE Distribution
              Domain = Default Domain
              Location = Hub to Shroud (mid-span)
              Variable = Turbulence Kinetic Energy
            END

            # Create efficiency expression
            EXPRESSION: eta_total
              Definition = torque_z()@Blade * {z_blades} * {omega:.6f} / \\
                (massFlowInt(Velocity, Inlet) * 9.80665 * \\
                 (massFlowAve(Total Pressure in Stn Frame)@Outlet \\
                  - massFlowAve(Total Pressure in Stn Frame)@Inlet) \\
                 / ({fluid_density:.1f} * 9.80665))
            END

            # Create head expression
            EXPRESSION: head_m
              Definition = (massFlowAve(Total Pressure in Stn Frame)@Outlet \\
                - massFlowAve(Total Pressure in Stn Frame)@Inlet) \\
                / ({fluid_density:.1f} * 9.80665)
            END

            # Create power expression
            EXPRESSION: shaft_power
              Definition = torque_z()@Blade * {z_blades} * {omega:.6f}
            END

            # Export key metrics to CSV
            TABLE: Performance Summary
              Export CSV = performance_summary.csv
              Column 1 = Efficiency
              Column 2 = Head Rise
              Column 3 = Power
              Column 4 = Torque
            END

            # Export blade loading data
            TABLE: Blade Loading Data
              Export CSV = blade_loading.csv
              Column 1 = Normalized Streamwise
              Column 2 = Pressure (Pressure Side)
              Column 3 = Pressure (Suction Side)
            END
        """)
        return template

    def generate_turbo_setup(self, sizing_result: SizingResult) -> dict[str, Any]:
        """Return TurboGrid setup parameters as a dictionary.

        These values can be fed into ANSYS TurboGrid for automated
        mesh generation of the impeller passage.

        Args:
            sizing_result: Output from hpe.sizing.

        Returns:
            Dictionary with TurboGrid machine definition parameters
            including machine_type, hub_rz, shroud_rz, blade_angles,
            n_blades, inlet_angle, outlet_angle.
        """
        mp = sizing_result.meridional_profile
        vt = sizing_result.velocity_triangles
        rpm = self._calc_rpm(sizing_result)

        d1 = sizing_result.impeller_d1
        d2 = sizing_result.impeller_d2
        b1 = mp.get("b1", sizing_result.impeller_b2 * 1.5)
        b2 = sizing_result.impeller_b2
        d1_hub = mp.get("d1_hub", d1 * 0.35)

        beta1_blade = sizing_result.beta1
        beta2_blade = sizing_result.beta2

        # Axial length estimate for meridional profile
        axial_length = d2 / 2.0 * 0.4

        # Hub R-Z coordinates (radius, axial): inlet -> turn -> outlet
        hub_rz = [
            {"r": d1_hub / 2.0, "z": 0.0},
            {"r": d1_hub / 2.0, "z": axial_length * 0.3},
            {"r": d2 / 2.0, "z": axial_length},
        ]

        # Shroud R-Z coordinates
        shroud_rz = [
            {"r": d1 / 2.0, "z": 0.0},
            {"r": d1 / 2.0, "z": axial_length * 0.3},
            {"r": d2 / 2.0, "z": axial_length},
        ]

        return {
            "machine_type": "Centrifugal Pump",
            "rotation_axis": "Z",
            "rpm": rpm,
            "n_blades": sizing_result.blade_count,
            "inlet_angle": beta1_blade,
            "outlet_angle": beta2_blade,
            "hub_rz": hub_rz,
            "shroud_rz": shroud_rz,
            "blade_angles": {
                "hub": {
                    "inlet_deg": beta1_blade * 0.95,
                    "outlet_deg": beta2_blade * 0.95,
                },
                "mid": {
                    "inlet_deg": beta1_blade,
                    "outlet_deg": beta2_blade,
                },
                "shroud": {
                    "inlet_deg": beta1_blade * 1.05,
                    "outlet_deg": beta2_blade * 1.05,
                },
            },
            "blade_profile": {
                "type": "circular_arc",
                "thickness_max_mm": max(2.0, d2 * 1000.0 * 0.02),
                "leading_edge_radius_mm": max(0.5, d2 * 1000.0 * 0.005),
                "trailing_edge_radius_mm": max(0.3, d2 * 1000.0 * 0.003),
            },
            "inlet_station": {
                "radius_m": d1 / 2.0,
                "width_m": b1,
                "hub_radius_m": d1_hub / 2.0,
            },
            "outlet_station": {
                "radius_m": d2 / 2.0,
                "width_m": b2,
            },
            "mesh_params": {
                "global_size_factor": 1.0,
                "boundary_layer_first_height_mm": 0.02,
                "boundary_layer_growth_ratio": 1.2,
                "boundary_layer_count": 15,
                "spanwise_elements": 30,
                "streamwise_elements": 80,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_roughness_block(roughness_um: float) -> str:
        """Build wall roughness CCL block.

        Args:
            roughness_um: Roughness in micrometers. 0 = smooth wall.

        Returns:
            Indented CCL lines for wall roughness.
        """
        if roughness_um > 0:
            return textwrap.dedent(f"""\
                WALL ROUGHNESS:
                  Option = Sand Grain
                  Sand Grain Roughness = {roughness_um:.2f} [um]
                END""")
        return textwrap.dedent("""\
                WALL ROUGHNESS:
                  Option = Smooth Wall
                END""")

    @staticmethod
    def _build_periodic_section() -> str:
        """Build periodic boundary CCL section for single-passage."""
        return textwrap.dedent("""\
            BOUNDARY: Periodic 1 Side 1
              Boundary Type = INTERFACE
              Location = periodic1side1
              BOUNDARY CONDITIONS:
                MASS AND MOMENTUM:
                  Option = Conservative Interface Flux
                END
              END
            END

            BOUNDARY: Periodic 1 Side 2
              Boundary Type = INTERFACE
              Location = periodic1side2
              BOUNDARY CONDITIONS:
                MASS AND MOMENTUM:
                  Option = Conservative Interface Flux
                END
              END
            END

            DOMAIN INTERFACE: Periodic 1
              Boundary List1 = Periodic 1 Side 1
              Boundary List2 = Periodic 1 Side 2
              Interface Type = Fluid Fluid
              INTERFACE MODELS:
                Option = Rotational Periodicity
                AXIS DEFINITION:
                  Option = Coordinate Axis
                  Rotation Axis = Z
                END
              END
            END""")

    @staticmethod
    def _calc_rpm(sizing: SizingResult) -> float:
        """Recover RPM from outlet velocity triangle."""
        u2 = sizing.velocity_triangles["outlet"]["u"]
        return 60.0 * u2 / (math.pi * sizing.impeller_d2)

    @staticmethod
    def _calc_mass_flow(
        sizing: SizingResult,
        rho: float,
    ) -> float:
        """Compute mass flow rate [kg/s] from velocity triangles."""
        vt_out = sizing.velocity_triangles["outlet"]
        cm2 = vt_out["cm"]
        d2 = sizing.impeller_d2
        b2 = sizing.impeller_b2
        q = math.pi * d2 * b2 * cm2  # m3/s
        return q * rho
