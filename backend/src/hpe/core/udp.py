"""User Defined Parameters (UDP) — extensible computed parameter system.

Allows users to register custom computed parameters that derive from
sizing results and operating conditions.  Ships with 10 built-in UDPs
covering the most common turbomachinery non-dimensional groups.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from hpe.core.models import OperatingPoint, SizingResult

G = 9.80665  # m/s²


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class UDPDefinition:
    """Metadata and compute function for a single UDP."""

    name: str
    compute_fn: Callable[[SizingResult, OperatingPoint], float]
    description: str
    unit: str
    category: str


@dataclass
class UDPResult:
    """Evaluated value of a single UDP."""

    value: float
    unit: str
    description: str


# ---------------------------------------------------------------------------
# Registry (thread-safe singleton)
# ---------------------------------------------------------------------------

class UDPRegistry:
    """Singleton registry for User Defined Parameters."""

    _instance: Optional[UDPRegistry] = None
    _lock = threading.Lock()

    def __new__(cls) -> UDPRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._params: Dict[str, UDPDefinition] = {}
                cls._instance._initialized = False
            return cls._instance

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def register(
        self,
        name: str,
        compute_fn: Callable[[SizingResult, OperatingPoint], float],
        description: str,
        unit: str,
        category: str,
    ) -> None:
        """Register a new UDP (or overwrite an existing one)."""
        self._params[name] = UDPDefinition(
            name=name,
            compute_fn=compute_fn,
            description=description,
            unit=unit,
            category=category,
        )

    def evaluate(
        self,
        name: str,
        sizing_result: SizingResult,
        op: OperatingPoint,
    ) -> float:
        """Compute a single UDP by name."""
        if name not in self._params:
            raise KeyError(f"UDP '{name}' not registered")
        return self._params[name].compute_fn(sizing_result, op)

    def evaluate_all(
        self,
        sizing_result: SizingResult,
        op: OperatingPoint,
    ) -> Dict[str, UDPResult]:
        """Compute all registered UDPs and return results."""
        results: Dict[str, UDPResult] = {}
        for name, defn in self._params.items():
            try:
                val = defn.compute_fn(sizing_result, op)
            except Exception:
                val = float("nan")
            results[name] = UDPResult(
                value=val,
                unit=defn.unit,
                description=defn.description,
            )
        return results

    def list_parameters(self) -> List[Dict[str, str]]:
        """Return metadata for all registered UDPs."""
        return [
            {
                "name": d.name,
                "description": d.description,
                "unit": d.unit,
                "category": d.category,
            }
            for d in self._params.values()
        ]

    def remove(self, name: str) -> None:
        """Unregister a UDP."""
        self._params.pop(name, None)

    def _ensure_builtins(self) -> None:
        """Register built-in UDPs if not already done."""
        if self._initialized:
            return
        self._initialized = True
        _register_builtins(self)


# ---------------------------------------------------------------------------
# Built-in UDPs
# ---------------------------------------------------------------------------

def _register_builtins(reg: UDPRegistry) -> None:
    """Register the 10 built-in non-dimensional parameters."""

    # 1. Peripheral speed at outlet
    def _peripheral_speed_u2(sr: SizingResult, op: OperatingPoint) -> float:
        return math.pi * sr.impeller_d2 * op.rpm / 60.0

    reg.register(
        "peripheral_speed_u2",
        _peripheral_speed_u2,
        "Peripheral (tip) speed at impeller outlet",
        "m/s",
        "velocity",
    )

    # 2. Suction specific speed
    def _suction_specific_speed(sr: SizingResult, op: OperatingPoint) -> float:
        npsh_r = sr.estimated_npsh_r
        if npsh_r <= 0:
            return float("nan")
        return op.rpm * math.sqrt(op.flow_rate) / (npsh_r ** 0.75)

    reg.register(
        "suction_specific_speed",
        _suction_specific_speed,
        "Suction specific speed Nss = n * Q^0.5 / NPSHr^0.75",
        "-",
        "specific_speed",
    )

    # 3. Head coefficient (psi)
    def _head_coefficient_psi(sr: SizingResult, op: OperatingPoint) -> float:
        u2 = math.pi * sr.impeller_d2 * op.rpm / 60.0
        if u2 == 0:
            return float("nan")
        return G * op.head / (u2 ** 2)

    reg.register(
        "head_coefficient_psi",
        _head_coefficient_psi,
        "Head coefficient psi = g*H / u2^2",
        "-",
        "coefficient",
    )

    # 4. Flow coefficient (phi)
    def _flow_coefficient_phi(sr: SizingResult, op: OperatingPoint) -> float:
        u2 = math.pi * sr.impeller_d2 * op.rpm / 60.0
        if u2 == 0:
            return float("nan")
        # cm2 from velocity triangles
        vt = sr.velocity_triangles_typed
        if vt and vt.outlet:
            cm2 = vt.outlet.cm
        else:
            # fallback estimate: Q / (pi * D2 * b2)
            area2 = math.pi * sr.impeller_d2 * sr.impeller_b2
            cm2 = op.flow_rate / area2 if area2 > 0 else 0.0
        return cm2 / u2

    reg.register(
        "flow_coefficient_phi",
        _flow_coefficient_phi,
        "Flow coefficient phi = cm2 / u2",
        "-",
        "coefficient",
    )

    # 5. Power specific speed
    def _power_specific_speed(sr: SizingResult, op: OperatingPoint) -> float:
        power = sr.estimated_power
        if power <= 0 or op.head <= 0:
            return float("nan")
        return op.rpm * math.sqrt(power) / (op.head ** 1.25)

    reg.register(
        "power_specific_speed",
        _power_specific_speed,
        "Power specific speed Ns_power = n * P^0.5 / H^1.25",
        "-",
        "specific_speed",
    )

    # 6. Tip speed ratio
    def _tip_speed_ratio(sr: SizingResult, op: OperatingPoint) -> float:
        u2 = math.pi * sr.impeller_d2 * op.rpm / 60.0
        denom = math.sqrt(2.0 * G * op.head)
        if denom == 0:
            return float("nan")
        return u2 / denom

    reg.register(
        "tip_speed_ratio",
        _tip_speed_ratio,
        "Tip speed ratio u2 / sqrt(2*g*H)",
        "-",
        "velocity",
    )

    # 7. Meridional velocity ratio
    def _meridional_velocity_ratio(sr: SizingResult, op: OperatingPoint) -> float:
        vt = sr.velocity_triangles_typed
        if vt and vt.inlet and vt.outlet:
            cm1 = vt.inlet.cm
            cm2 = vt.outlet.cm
            if cm1 == 0:
                return float("nan")
            return cm2 / cm1
        return float("nan")

    reg.register(
        "meridional_velocity_ratio",
        _meridional_velocity_ratio,
        "Meridional velocity ratio cm2 / cm1",
        "-",
        "velocity",
    )

    # 8. Blade solidity (approximate at mean radius)
    def _blade_solidity(sr: SizingResult, op: OperatingPoint) -> float:
        # Approximate chord from meridional length / cos(mean beta)
        d1 = sr.impeller_d1
        d2 = sr.impeller_d2
        b2 = sr.impeller_b2
        z = sr.blade_count
        r_mean = (d1 / 2.0 + d2 / 2.0) / 2.0
        pitch = 2.0 * math.pi * r_mean / z if z > 0 else float("inf")
        # Rough chord estimate: sqrt((d2/2 - d1/2)^2 + b2^2)
        chord = math.sqrt((d2 / 2.0 - d1 / 2.0) ** 2 + b2 ** 2)
        if pitch == 0:
            return float("nan")
        return chord / pitch

    reg.register(
        "blade_solidity",
        _blade_solidity,
        "Blade solidity (chord / pitch) at mean radius",
        "-",
        "geometry",
    )

    # 9. Zweifel coefficient
    def _zweifel_coefficient(sr: SizingResult, op: OperatingPoint) -> float:
        vt = sr.velocity_triangles_typed
        if not (vt and vt.inlet and vt.outlet):
            return float("nan")
        alpha1_rad = math.radians(vt.inlet.alpha)
        alpha2_rad = math.radians(vt.outlet.alpha)
        cos2 = math.cos(alpha2_rad) ** 2
        if cos2 == 0:
            return float("nan")
        # s/c from blade_solidity inverse
        d1 = sr.impeller_d1
        d2 = sr.impeller_d2
        b2 = sr.impeller_b2
        z = sr.blade_count
        r_mean = (d1 / 2.0 + d2 / 2.0) / 2.0
        pitch = 2.0 * math.pi * r_mean / z if z > 0 else 0.0
        chord = math.sqrt((d2 / 2.0 - d1 / 2.0) ** 2 + b2 ** 2)
        if chord == 0:
            return float("nan")
        s_over_c = pitch / chord
        return 2.0 * s_over_c * cos2 * (
            math.tan(alpha1_rad) - math.tan(alpha2_rad)
        )

    reg.register(
        "zweifel_coefficient",
        _zweifel_coefficient,
        "Zweifel coefficient Cz = 2*(s/c)*cos^2(alpha2)*(tan(alpha1)-tan(alpha2))",
        "-",
        "loading",
    )

    # 10. Cordier diameter
    def _cordier_diameter(sr: SizingResult, op: OperatingPoint) -> float:
        if op.head <= 0 or op.flow_rate <= 0:
            return float("nan")
        return sr.impeller_d2 * (G * op.head) ** 0.25 / math.sqrt(op.flow_rate)

    reg.register(
        "cordier_diameter",
        _cordier_diameter,
        "Cordier specific diameter delta = D2 * (g*H)^0.25 / Q^0.5",
        "-",
        "specific_speed",
    )


# ---------------------------------------------------------------------------
# Module-level helper: get the global registry with builtins loaded
# ---------------------------------------------------------------------------

def get_registry() -> UDPRegistry:
    """Return the singleton UDPRegistry with built-in UDPs loaded."""
    reg = UDPRegistry()
    reg._ensure_builtins()
    return reg
