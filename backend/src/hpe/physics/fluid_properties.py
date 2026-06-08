"""Unified fluid properties interface for the HPE sizing pipeline.

Provides a single FluidProperties class that can represent both
incompressible (water, oil) and compressible (air, gases, refrigerants)
working fluids.  Predefined instances cover the most common fluids.

The class bridges the gap between the incompressible pump sizing
(which uses rho, mu) and the compressible turbomachinery path
(which needs gamma, cp, R, total conditions).

Usage:
    from hpe.physics.fluid_properties import FluidProperties, WATER, AIR

    # Incompressible
    op = OperatingPoint(..., fluid_props=WATER)

    # Compressible from ideal-gas law
    props = FluidProperties.from_ideal_gas(gamma=1.4, R=287.05, T=300, p=101325)

    # Real-gas table lookup
    props = FluidProperties.from_real_gas_table("R134a", T=300, p=500e3)

References:
    - Cengel & Cimbala (2018). Fluid Mechanics.
    - Lemmon et al. (2010). NIST REFPROP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass



@dataclass
class FluidProperties:
    """Working-fluid thermophysical properties at a reference state.

    Attributes:
        name: Human-readable fluid identifier.
        rho: Density [kg/m^3].
        mu: Dynamic viscosity [Pa s].
        gamma: Ratio of specific heats Cp/Cv (1.0 for incompressible).
        cp: Specific heat at constant pressure [J/(kg K)].
        R_specific: Specific gas constant [J/(kg K)].  Zero for liquids.
        p_vapor: Vapour pressure at the reference temperature [Pa].
    """

    name: str
    rho: float
    mu: float
    gamma: float
    cp: float
    R_specific: float
    p_vapor: float

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_ideal_gas(
        cls,
        gamma: float,
        R: float,
        T: float,
        p: float,
        name: str = "IdealGas",
    ) -> FluidProperties:
        """Create fluid properties from ideal-gas parameters.

        Computes density, Cp, and sets viscosity to Sutherland's law
        for air-like gases.

        Args:
            gamma: Cp/Cv ratio.
            R: Specific gas constant [J/(kg K)].
            T: Temperature [K].
            p: Pressure [Pa].
            name: Fluid name tag.

        Returns:
            FluidProperties instance.
        """
        rho = p / (R * T) if T > 0 else 1.0
        cp = gamma * R / (gamma - 1.0) if gamma > 1.0 else R
        # Sutherland viscosity (air-like)
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4
        mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
        return cls(
            name=name,
            rho=rho,
            mu=mu,
            gamma=gamma,
            cp=cp,
            R_specific=R,
            p_vapor=0.0,
        )

    @classmethod
    def from_real_gas_table(
        cls,
        fluid_name: str,
        T: float,
        p: float,
    ) -> FluidProperties:
        """Interpolate properties from built-in tables or real_gas module.

        Falls back to the corresponding-states model in
        ``hpe.physics.real_gas`` when CoolProp is unavailable.

        Args:
            fluid_name: Fluid identifier (e.g. ``"R134a"``, ``"CO2"``).
            T: Temperature [K].
            p: Pressure [Pa].

        Returns:
            FluidProperties instance at the given state.
        """
        from hpe.physics.real_gas import get_state, FLUIDS

        state = get_state(fluid_name, T, p)

        # Derive specific gas constant
        fluid_data = FLUIDS.get(fluid_name)
        R_spec = 8.314 / fluid_data.molar_mass if fluid_data else 0.0

        # Rough viscosity from kinetic theory / empirical
        mu = _estimate_viscosity(fluid_name, T, state.rho)

        # Vapour pressure estimate (Antoine-like, rough)
        p_vap = _estimate_p_vapor(fluid_name, T)

        return cls(
            name=fluid_name,
            rho=state.rho,
            mu=mu,
            gamma=state.gamma,
            cp=state.cp,
            R_specific=R_spec,
            p_vapor=p_vap,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_compressible(self, mach_threshold: float = 0.3) -> bool:
        """Return True if the fluid behaves as a compressible gas.

        A fluid is treated as compressible when its specific gas constant
        is positive (i.e. it is a gas, not a liquid) **and** typical
        velocities could exceed the Mach threshold.  For the sizing
        pipeline the simple heuristic is R_specific > 0.

        Args:
            mach_threshold: Mach number above which compressibility
                effects are significant (default 0.3).

        Returns:
            True for gaseous fluids.
        """
        return self.R_specific > 1.0 and self.gamma > 1.0

    def compressibility_factor(self, T: float, p: float) -> float:
        """Compute the compressibility factor Z = p / (rho R T).

        Z == 1.0 for an ideal gas; deviations indicate real-gas effects.

        Args:
            T: Temperature [K].
            p: Pressure [Pa].

        Returns:
            Compressibility factor Z (dimensionless).
        """
        if self.R_specific <= 0 or T <= 0:
            return 1.0
        rho_ideal = p / (self.R_specific * T)
        if rho_ideal <= 0:
            return 1.0
        return self.rho / rho_ideal if rho_ideal != 0 else 1.0

    def speed_of_sound(self, T: float | None = None) -> float:
        """Speed of sound [m/s].

        For gases: a = sqrt(gamma * R * T).
        For liquids: returns ~1480 m/s (water-like default).

        Args:
            T: Temperature [K].  Uses rho-derived T if omitted.

        Returns:
            Speed of sound [m/s].
        """
        if self.R_specific > 1.0 and self.gamma > 1.0:
            T_eff = T if T is not None else (
                self.rho * self.R_specific  # p/R  (rough)
            )
            T_eff = max(T_eff, 1.0)
            return math.sqrt(self.gamma * self.R_specific * T_eff)
        # Incompressible default (water)
        return 1480.0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FluidProperties(name={self.name!r}, rho={self.rho:.3f}, "
            f"mu={self.mu:.3e}, gamma={self.gamma:.3f})"
        )


# ======================================================================
# Predefined fluids
# ======================================================================

WATER = FluidProperties(
    name="Water",
    rho=998.2,
    mu=1.003e-3,
    gamma=1.0,
    cp=4182.0,
    R_specific=0.0,
    p_vapor=2339.0,  # at 20 C
)

AIR = FluidProperties(
    name="Air",
    rho=1.204,
    mu=1.825e-5,
    gamma=1.4,
    cp=1004.5,
    R_specific=287.05,
    p_vapor=0.0,
)

R134A = FluidProperties(
    name="R134a",
    rho=1206.0,  # liquid at 25 C
    mu=2.01e-4,
    gamma=1.127,
    cp=865.0,
    R_specific=81.49,
    p_vapor=665_800.0,  # at 25 C
)

CO2 = FluidProperties(
    name="CO2",
    rho=1.842,  # gas at 25 C, 101325 Pa
    mu=1.48e-5,
    gamma=1.289,
    cp=844.0,
    R_specific=188.92,
    p_vapor=0.0,
)

NATURAL_GAS = FluidProperties(
    name="NaturalGas",
    rho=0.72,  # mostly methane at STP
    mu=1.1e-5,
    gamma=1.31,
    cp=2190.0,
    R_specific=518.3,  # methane-dominant
    p_vapor=0.0,
)

STEAM = FluidProperties(
    name="Steam",
    rho=0.590,  # at 100 C, 101325 Pa
    mu=1.2e-5,
    gamma=1.33,
    cp=2080.0,
    R_specific=461.5,
    p_vapor=101325.0,  # at 100 C (by definition)
)


# Name-based lookup
FLUID_LIBRARY: dict[str, FluidProperties] = {
    "water": WATER,
    "air": AIR,
    "r134a": R134A,
    "co2": CO2,
    "natural_gas": NATURAL_GAS,
    "steam": STEAM,
}


def get_fluid(name: str) -> FluidProperties:
    """Look up a predefined fluid by name (case-insensitive).

    Args:
        name: Fluid name (e.g. ``"water"``, ``"Air"``, ``"CO2"``).

    Returns:
        FluidProperties instance.

    Raises:
        KeyError: If the fluid is not in the library.
    """
    key = name.strip().lower()
    if key not in FLUID_LIBRARY:
        raise KeyError(
            f"Unknown fluid {name!r}. Available: {list(FLUID_LIBRARY.keys())}"
        )
    return FLUID_LIBRARY[key]


# ======================================================================
# Internal helpers
# ======================================================================

def _estimate_viscosity(fluid_name: str, T: float, rho: float) -> float:
    """Rough viscosity estimate for common refrigerants/gases.

    Uses simple power-law or Sutherland-like models.
    """
    # Gas-phase viscosity estimates [Pa s] at ~ 300 K
    _mu_gas: dict[str, float] = {
        "Air": 1.82e-5,
        "CO2": 1.48e-5,
        "R134a": 1.16e-5,
        "R410A": 1.25e-5,
        "R245fa": 1.0e-5,
        "Water": 1.0e-5,  # steam
    }
    mu_ref = _mu_gas.get(fluid_name, 1.5e-5)
    # Simple power-law scaling with temperature
    T_ref = 300.0
    if T > 0:
        return mu_ref * (T / T_ref) ** 0.7
    return mu_ref


def _estimate_p_vapor(fluid_name: str, T: float) -> float:
    """Rough vapour-pressure estimate [Pa] using Clausius-Clapeyron-like form."""
    # Reference boiling points at 101325 Pa
    _t_boil: dict[str, float] = {
        "Water": 373.15,
        "R134a": 247.08,
        "R410A": 221.71,
        "CO2": 194.65,  # sublimation
        "R245fa": 288.29,
        "Air": 78.8,  # N2 dominant
    }
    t_b = _t_boil.get(fluid_name, 300.0)
    # Clausius-Clapeyron (simplified)
    if T <= 0 or t_b <= 0:
        return 0.0
    p_ref = 101325.0
    # L/R ~ 10*T_boil (rough universal correlation)
    L_over_R = 10.0 * t_b
    p_vap = p_ref * math.exp(L_over_R * (1.0 / t_b - 1.0 / T))
    return max(0.0, p_vap)
