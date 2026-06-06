"""DesignState — a single, coherent design object that flows through every stage.

Inspired by LEAP 71's "computational engineering model" idea: instead of
passing loose dicts between sizing → geometry → physics → CFD and converting
them at each route boundary, a ``DesignState`` carries:

* the **intent** — the operating point plus the engineering ``DesignConstraints``
  the design must satisfy (De Haller, tip speed, NPSH, Nq range, ...);
* the **accumulated results** of each stage that has run;
* **provenance** — which stages ran, in order, and whether they succeeded;
* a built-in **constraint evaluation** so any orchestrator can ask "is this
  design feasible?" without re-deriving the rules.

This module is intentionally dependency-light (pure dataclasses + the existing
``PhysicsValidator``) and **additive**: it does not replace ``SizingResult`` or
``orchestrator.versions.DesignVersion`` — it composes their data. It is the
foundation the constraint-driven design facade (the next step) will iterate on.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from hpe.core.enums import DesignMode

# ---------------------------------------------------------------------------
# Constraints (the "intent" the design must satisfy)
# ---------------------------------------------------------------------------

@dataclass
class DesignConstraints:
    """Engineering limits a feasible design must respect.

    All bounds are optional: ``None`` disables that particular check.
    Defaults reflect common centrifugal-pump practice (Gülich).
    """

    min_efficiency: Optional[float] = 0.70      # total efficiency [-]
    max_tip_speed: Optional[float] = 45.0       # u2 [m/s] (cast-iron/material limit)
    min_de_haller: Optional[float] = 0.70       # w2/w1 diffusion ratio [-]
    max_npsh_r: Optional[float] = None          # required NPSH [m]
    nq_min: Optional[float] = 5.0               # specific speed lower bound
    nq_max: Optional[float] = 250.0             # specific speed upper bound
    max_sigma: Optional[float] = None           # Thoma cavitation index [-]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConstraintCheck:
    """Result of a single constraint evaluation."""

    name: str
    ok: bool
    value: Optional[float]
    limit: str
    message: str


@dataclass
class ConstraintReport:
    """Aggregate feasibility verdict for a design."""

    feasible: bool
    checks: list[ConstraintCheck] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)     # hard input errors
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[ConstraintCheck]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "checks": [asdict(c) for c in self.checks],
            "failed": [c.name for c in self.failed],
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """One entry in the design's audit trail."""

    stage: str
    status: str = "ok"   # ok | error | skipped
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# DesignState
# ---------------------------------------------------------------------------

# Stages whose output the state accumulates, in canonical pipeline order.
STAGES = ("sizing", "geometry", "physics", "surrogate", "cfd")


@dataclass
class DesignState:
    """A complete design in progress: intent + accumulated stage results."""

    operating_point: dict
    constraints: DesignConstraints = field(default_factory=DesignConstraints)
    mode: DesignMode = DesignMode.CLASSIC

    # Accumulated stage outputs (each a plain dict for serialisation).
    sizing: Optional[dict] = None
    geometry: Optional[dict] = None
    physics: Optional[dict] = None
    surrogate: Optional[dict] = None
    cfd: Optional[dict] = None

    provenance: list[StageRecord] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_operating_point(
        cls,
        op: Any,
        constraints: Optional[DesignConstraints] = None,
        mode: DesignMode = DesignMode.CLASSIC,
    ) -> "DesignState":
        """Build a fresh state from an OperatingPoint (object or dict)."""
        return cls(
            operating_point=_to_op_dict(op),
            constraints=constraints or DesignConstraints(),
            mode=mode,
        )

    # -- recording stage results ------------------------------------------

    def record_stage(
        self,
        stage: str,
        result: Any,
        status: str = "ok",
        note: str = "",
    ) -> "DesignState":
        """Attach a stage result and append to the provenance trail.

        ``stage`` must be one of :data:`STAGES`. ``result`` may be a dict or
        any object exposing ``to_dict()`` / being dataclass-serialisable.
        """
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
        setattr(self, stage, _to_dict(result))
        self.provenance.append(StageRecord(stage=stage, status=status, note=note))
        return self

    @property
    def stages_run(self) -> list[str]:
        return [s for s in STAGES if getattr(self, s) is not None]

    # -- feasibility -------------------------------------------------------

    def evaluate_constraints(self) -> ConstraintReport:
        """Check the current design against its constraints.

        Combines hard input validation (``PhysicsValidator``) with the
        engineering checks derivable from the sizing result. Checks are only
        emitted when both the constraint bound and the required value exist.
        """
        op = self.operating_point
        checks: list[ConstraintCheck] = []
        errors: list[str] = []
        warnings: list[str] = []

        # Hard input validity (Q, H, n within physical bounds).
        try:
            from hpe.sizing.validator import PhysicsValidator
            vr = PhysicsValidator.validate(
                float(op.get("flow_rate", 0.0)),
                float(op.get("head", 0.0)),
                float(op.get("rpm", 0.0)),
            )
            errors.extend(vr.errors)
            warnings.extend(vr.warnings)
        except Exception:  # noqa: BLE001 — validator must not break feasibility
            pass

        c = self.constraints
        s = self.sizing or {}

        if s:
            eta = _num(s.get("estimated_efficiency"))
            npsh = _num(s.get("estimated_npsh_r"))
            nq = _num(s.get("specific_speed_nq"))
            sigma = _num(s.get("sigma"))
            de_haller = _num(s.get("diffusion_ratio"))
            u2 = self._tip_speed()

            if c.min_efficiency is not None and eta is not None:
                checks.append(_check_min("min_efficiency", eta, c.min_efficiency, "η"))
            if c.max_tip_speed is not None and u2 is not None:
                checks.append(_check_max("max_tip_speed", u2, c.max_tip_speed, "u₂ [m/s]"))
            if c.min_de_haller is not None and de_haller:
                checks.append(_check_min("min_de_haller", de_haller, c.min_de_haller, "De Haller"))
            if c.max_npsh_r is not None and npsh is not None:
                checks.append(_check_max("max_npsh_r", npsh, c.max_npsh_r, "NPSHr [m]"))
            if c.max_sigma is not None and sigma is not None:
                checks.append(_check_max("max_sigma", sigma, c.max_sigma, "σ"))
            if nq is not None:
                if c.nq_min is not None:
                    checks.append(_check_min("nq_min", nq, c.nq_min, "Nq"))
                if c.nq_max is not None:
                    checks.append(_check_max("nq_max", nq, c.nq_max, "Nq"))

        feasible = not errors and all(chk.ok for chk in checks)
        return ConstraintReport(feasible=feasible, checks=checks, errors=errors, warnings=warnings)

    @property
    def is_feasible(self) -> bool:
        return self.evaluate_constraints().feasible

    def _tip_speed(self) -> Optional[float]:
        """Outlet tip speed u₂ from sizing, or computed from D2 and rpm."""
        s = self.sizing or {}
        vt = s.get("velocity_triangles") or {}
        outlet = vt.get("outlet") if isinstance(vt, dict) else None
        if isinstance(outlet, dict) and outlet.get("u"):
            return _num(outlet.get("u"))
        d2 = _num(s.get("impeller_d2"))
        rpm = _num(self.operating_point.get("rpm"))
        if d2 and rpm:
            return math.pi * d2 * rpm / 60.0
        return None

    # -- views / serialisation --------------------------------------------

    def summary(self) -> dict:
        op = self.operating_point
        s = self.sizing or {}
        report = self.evaluate_constraints()
        return {
            "id": self.id,
            "mode": self.mode.value,
            "operating_point": {
                "Q_m3h": round(_num(op.get("flow_rate", 0.0)) * 3600, 2),
                "H_m": _num(op.get("head")),
                "rpm": _num(op.get("rpm")),
            },
            "stages_run": self.stages_run,
            "nq": _num(s.get("specific_speed_nq")),
            "eta": _num(s.get("estimated_efficiency")),
            "feasible": report.feasible,
            "failed_constraints": [c.name for c in report.failed],
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode.value,
            "operating_point": self.operating_point,
            "constraints": self.constraints.to_dict(),
            "sizing": self.sizing,
            "geometry": self.geometry,
            "physics": self.physics,
            "surrogate": self.surrogate,
            "cfd": self.cfd,
            "provenance": [p.to_dict() for p in self.provenance],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DesignState":
        return cls(
            operating_point=data.get("operating_point", {}),
            constraints=DesignConstraints(**(data.get("constraints") or {})),
            mode=DesignMode(data.get("mode", DesignMode.CLASSIC.value)),
            sizing=data.get("sizing"),
            geometry=data.get("geometry"),
            physics=data.get("physics"),
            surrogate=data.get("surrogate"),
            cfd=data.get("cfd"),
            provenance=[StageRecord(**p) for p in data.get("provenance", [])],
            id=data.get("id", uuid.uuid4().hex),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_min(name: str, value: float, limit: float, label: str) -> ConstraintCheck:
    ok = value >= limit
    return ConstraintCheck(
        name=name, ok=ok, value=round(value, 4), limit=f">= {limit}",
        message=f"{label}={value:.3g} {'OK' if ok else f'< mínimo {limit:g}'}",
    )


def _check_max(name: str, value: float, limit: float, label: str) -> ConstraintCheck:
    ok = value <= limit
    return ConstraintCheck(
        name=name, ok=ok, value=round(value, 4), limit=f"<= {limit}",
        message=f"{label}={value:.3g} {'OK' if ok else f'> máximo {limit:g}'}",
    )


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_op_dict(op: Any) -> dict:
    """Normalise an OperatingPoint (object or dict) to a plain dict."""
    if isinstance(op, dict):
        return dict(op)
    out = {}
    for k in ("flow_rate", "head", "rpm", "machine_type", "fluid",
              "fluid_density", "fluid_viscosity"):
        if hasattr(op, k):
            val = getattr(op, k)
            out[k] = getattr(val, "value", val)  # unwrap enums
    return out


def _to_dict(result: Any) -> dict:
    """Best-effort conversion of a stage result to a serialisable dict."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return result.to_dict()
    try:
        return asdict(result)
    except TypeError:
        # Fall back to public attributes.
        return {k: v for k, v in vars(result).items() if not k.startswith("_")}
