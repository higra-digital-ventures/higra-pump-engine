"""CFD-in-the-design-loop — automated sizing → geometry → mesh → solve → compare.

Iteratively runs CFD simulations and compares results against design
targets (head, efficiency).  Geometry is updated each iteration until
convergence or a budget is exhausted.

References:
    Denton (1997) — Loss mechanisms in turbomachinery.
    Gulich (2014) — Centrifugal Pumps, Ch. 8 (CFD validation).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


from hpe.cfd.openfoam.case_builder import build_case
from hpe.cfd.openfoam.runner import run_case, RunResult
from hpe.core.models import SizingResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CFDResults:
    """Post-processed results extracted from a single CFD run."""

    head: float  # Total head rise [m]
    efficiency: float  # Hydraulic efficiency [-]
    power: float  # Shaft power [W]
    pressure_rise: float  # Total-to-total pressure rise [Pa]
    total_pressure_loss: float  # Total pressure loss [Pa]
    mass_flow_check: float  # Continuity residual (should be ~0)
    blade_loading: list[dict[str, Any]]  # Cp vs chord at each span
    convergence_residuals: list[float]  # Final residual history


@dataclass
class DesignLoopIteration:
    """Record for one iteration of the design loop."""

    iteration: int
    head_target: float
    head_cfd: float
    eta_target: float
    eta_cfd: float
    geometry_changes: dict[str, float]


@dataclass
class DesignLoopResult:
    """Full result of the automated CFD design loop."""

    converged: bool
    n_iterations: int
    final_results: CFDResults
    history: list[DesignLoopIteration]
    run_id: str


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CFDDesignLoop:
    """Automated CFD design loop for centrifugal pump impellers.

    Workflow:
        1. Sizing (meanline) produces initial geometry.
        2. OpenFOAM case is built and executed.
        3. Results are extracted and compared with targets.
        4. Geometry is adjusted (beta2, D2 trim) and the loop repeats.
    """

    def __init__(
        self,
        work_dir: Path | str = Path("cfd_runs"),
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._sizing: Optional[SizingResult] = None
        self._mesh_params: dict[str, Any] = {}
        self._solver_params: dict[str, Any] = {}
        self._step_file: Optional[Path] = None

        self._target_head: float = 0.0
        self._target_efficiency: float = 0.0

        self._history: list[DesignLoopIteration] = []
        self._run_id: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(
        self,
        sizing_result: SizingResult,
        mesh_params: dict[str, Any] | None = None,
        solver_params: dict[str, Any] | None = None,
        step_file: Path | str | None = None,
        target_head: float | None = None,
        target_efficiency: float | None = None,
    ) -> None:
        """Prepare the design loop with initial sizing and configuration.

        Args:
            sizing_result: Result from meanline sizing.
            mesh_params: Optional mesh overrides (e.g. refinement level).
            solver_params: Optional solver overrides (e.g. n_procs, timeout).
            step_file: Path to STEP geometry.  If ``None`` a placeholder
                       is generated.
            target_head: Design head target [m].  Falls back to sizing head.
            target_efficiency: Design efficiency target [-].
        """
        self._sizing = sizing_result
        self._mesh_params = mesh_params or {}
        self._solver_params = solver_params or {}
        self._step_file = Path(step_file) if step_file else None
        self._run_id = uuid.uuid4().hex[:12]

        # Targets
        vt = sizing_result.velocity_triangles
        if isinstance(vt, dict):
            euler_head = vt.get("euler_head", sizing_result.head if hasattr(sizing_result, "head") else 0.0)
        else:
            euler_head = vt.euler_head if hasattr(vt, "euler_head") else 0.0

        self._target_head = target_head if target_head is not None else euler_head
        self._target_efficiency = target_efficiency if target_efficiency is not None else sizing_result.estimated_efficiency

        self._history = []

    def run_single(self, case_dir: Path | str) -> list[RunResult]:
        """Execute one CFD run using the OpenFOAM runner.

        Args:
            case_dir: Path to a prepared OpenFOAM case directory.

        Returns:
            List of :class:`RunResult` from each solver step.
        """
        case_dir = Path(case_dir)
        n_procs = self._solver_params.get("n_procs", 4)
        timeout = self._solver_params.get("timeout", None)
        parallel = n_procs > 1

        return run_case(
            case_dir=case_dir,
            parallel=parallel,
            n_procs=n_procs,
            timeout=timeout,
        )

    def extract_results(self, case_dir: Path | str) -> CFDResults:
        """Extract CFD results from a completed case directory.

        Reads the final time-step in *case_dir* and computes head,
        efficiency, power, pressure rise, losses, continuity check,
        blade loading, and convergence residuals.

        Args:
            case_dir: Path to a completed OpenFOAM case.

        Returns:
            :class:`CFDResults` dataclass.
        """
        case_dir = Path(case_dir)
        rho = 998.2  # water density [kg/m3]
        g = 9.80665

        # --- Parse pressure / velocity fields from the latest time directory ---
        pressure_rise, total_pressure_loss = _parse_pressure_fields(case_dir, rho)

        # Head from pressure rise
        head_cfd = pressure_rise / (rho * g) if rho * g > 0 else 0.0

        # Flow rate from boundary patches
        mass_flow = _parse_mass_flow(case_dir)
        volume_flow = abs(mass_flow) / rho if rho > 0 else 0.0

        # Power = rho * g * Q * H / eta  →  but we compute from torque here
        power = _parse_torque_power(case_dir, self._sizing)

        # Efficiency
        useful_power = rho * g * volume_flow * head_cfd
        efficiency = useful_power / power if power > 0 else 0.0

        # Blade loading
        blade_loading = _parse_blade_loading(case_dir)

        # Convergence residuals
        residuals = _parse_residuals(case_dir)

        # Continuity residual (last value)
        mass_flow_check = residuals[-1] if residuals else 0.0

        return CFDResults(
            head=head_cfd,
            efficiency=min(efficiency, 1.0),
            power=power,
            pressure_rise=pressure_rise,
            total_pressure_loss=total_pressure_loss,
            mass_flow_check=mass_flow_check,
            blade_loading=blade_loading,
            convergence_residuals=residuals,
        )

    def compare(
        self,
        cfd_results: CFDResults,
        target_head: float | None = None,
        target_efficiency: float | None = None,
    ) -> dict[str, float]:
        """Compare CFD results against design targets.

        Args:
            cfd_results: Results from :meth:`extract_results`.
            target_head: Override head target [m].
            target_efficiency: Override efficiency target [-].

        Returns:
            Dictionary with absolute and relative deltas.
        """
        t_head = target_head if target_head is not None else self._target_head
        t_eta = target_efficiency if target_efficiency is not None else self._target_efficiency

        delta_head = cfd_results.head - t_head
        delta_eta = cfd_results.efficiency - t_eta

        rel_head = delta_head / t_head if abs(t_head) > 1e-9 else 0.0
        rel_eta = delta_eta / t_eta if abs(t_eta) > 1e-9 else 0.0

        return {
            "delta_head_m": delta_head,
            "delta_head_rel": rel_head,
            "delta_efficiency": delta_eta,
            "delta_efficiency_rel": rel_eta,
            "head_converged": abs(rel_head) < 0.02,
            "eta_converged": abs(rel_eta) < 0.01,
        }

    def run_design_loop(
        self,
        max_iterations: int = 5,
        head_tolerance: float = 0.02,
        eta_tolerance: float = 0.01,
    ) -> DesignLoopResult:
        """Run the full automated design loop.

        Iterates: build case → solve → extract → compare → adjust geometry.
        Stops when both head and efficiency are within tolerance or the
        iteration budget is exhausted.

        Args:
            max_iterations: Maximum number of CFD evaluations.
            head_tolerance: Relative head convergence tolerance.
            eta_tolerance: Absolute efficiency convergence tolerance.

        Returns:
            :class:`DesignLoopResult` with convergence status and history.
        """
        if self._sizing is None:
            raise RuntimeError("Call setup() before run_design_loop().")

        converged = False
        final_results: Optional[CFDResults] = None

        for iteration in range(1, max_iterations + 1):
            # 1. Build case
            case_dir = self.work_dir / f"iter_{iteration:03d}"
            step_file = self._step_file or (case_dir / "placeholder.step")

            build_case(
                sizing=self._sizing,
                step_file=step_file,
                output_dir=case_dir,
                n_procs=self._solver_params.get("n_procs", 4),
            )

            # 2. Run CFD
            run_results = self.run_single(case_dir)

            # Check if solver succeeded
            if not all(r.success for r in run_results):
                # Record failed iteration and continue
                self._history.append(DesignLoopIteration(
                    iteration=iteration,
                    head_target=self._target_head,
                    head_cfd=0.0,
                    eta_target=self._target_efficiency,
                    eta_cfd=0.0,
                    geometry_changes={},
                ))
                continue

            # 3. Extract results
            cfd_results = self.extract_results(case_dir)
            final_results = cfd_results

            # 4. Compare
            deltas = self.compare(cfd_results)

            # 5. Compute geometry adjustments
            geometry_changes = _compute_geometry_adjustment(
                deltas, self._sizing,
            )

            # 6. Record
            self._history.append(DesignLoopIteration(
                iteration=iteration,
                head_target=self._target_head,
                head_cfd=cfd_results.head,
                eta_target=self._target_efficiency,
                eta_cfd=cfd_results.efficiency,
                geometry_changes=geometry_changes,
            ))

            # 7. Check convergence
            head_ok = abs(deltas["delta_head_rel"]) <= head_tolerance
            eta_ok = abs(deltas["delta_efficiency"]) <= eta_tolerance

            if head_ok and eta_ok:
                converged = True
                break

            # 8. Apply geometry adjustments for next iteration
            self._sizing = _apply_geometry_changes(self._sizing, geometry_changes)

        if final_results is None:
            final_results = CFDResults(
                head=0.0, efficiency=0.0, power=0.0,
                pressure_rise=0.0, total_pressure_loss=0.0,
                mass_flow_check=0.0, blade_loading=[], convergence_residuals=[],
            )

        return DesignLoopResult(
            converged=converged,
            n_iterations=len(self._history),
            final_results=final_results,
            history=self._history,
            run_id=self._run_id,
        )


# ---------------------------------------------------------------------------
# Private helpers — post-processing parsers
# ---------------------------------------------------------------------------

def _find_latest_time_dir(case_dir: Path) -> Path | None:
    """Find the latest numerical time directory in the case."""
    time_dirs: list[tuple[float, Path]] = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                time_dirs.append((t, d))
            except ValueError:
                continue
    if not time_dirs:
        return None
    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]


def _parse_pressure_fields(case_dir: Path, rho: float) -> tuple[float, float]:
    """Parse pressure rise and total pressure loss from post-processing files.

    Falls back to estimating from log files if field files are unavailable.

    Returns:
        (pressure_rise [Pa], total_pressure_loss [Pa])
    """
    # Try reading from postProcessing/
    pp_dir = case_dir / "postProcessing"
    if pp_dir.exists():
        # Look for fieldAverage or patchAverage
        for sub in pp_dir.iterdir():
            if "inlet" in sub.name.lower() or "outlet" in sub.name.lower():
                # Try to read the latest data file
                data_files = sorted(sub.glob("**/p*"), key=lambda f: f.stat().st_mtime)
                if data_files:
                    try:
                        lines = data_files[-1].read_text().strip().splitlines()
                        values = [float(v) for v in lines[-1].split() if _is_float(v)]
                        if len(values) >= 2:
                            p_in, p_out = values[0], values[-1]
                            pressure_rise = p_out - p_in
                            return pressure_rise, max(0.0, -pressure_rise * 0.05)
                    except (ValueError, IndexError):
                        pass

    # Fallback: parse simpleFoam log for pressure info
    log_file = case_dir / "log.simpleFoam"
    if not log_file.exists():
        # Try alternative log locations
        for candidate in ["log.simpleFoam", "log", "simpleFoam.log"]:
            lf = case_dir / candidate
            if lf.exists():
                log_file = lf
                break

    if log_file.exists():
        try:
            content = log_file.read_text()
            # Parse last pressure residual as proxy
            lines = content.splitlines()
            for line in reversed(lines):
                if "Solving for p" in line and "Final residual" in line:
                    parts = line.split("Final residual = ")
                    if len(parts) > 1:
                        residual = float(parts[1].split(",")[0])
                        # Very rough estimate — 50 m head pump at 998 kg/m3
                        estimated_rise = rho * 9.80665 * 50.0 * (1.0 - residual)
                        return estimated_rise, estimated_rise * 0.05
                    break
        except (ValueError, OSError):
            pass

    return 0.0, 0.0


def _parse_mass_flow(case_dir: Path) -> float:
    """Parse mass flow rate from postProcessing or log files.

    Returns:
        Mass flow rate [kg/s] (positive = into domain).
    """
    pp_flow = case_dir / "postProcessing" / "flowRatePatch"
    if pp_flow.exists():
        data_files = sorted(pp_flow.glob("**/surfaceFieldValue*"), key=lambda f: f.name)
        if data_files:
            try:
                last_line = data_files[-1].read_text().strip().splitlines()[-1]
                return abs(float(last_line.split()[-1]))
            except (ValueError, IndexError):
                pass
    return 0.0


def _parse_torque_power(case_dir: Path, sizing: Optional[SizingResult]) -> float:
    """Parse torque from forces postProcessing or estimate from sizing.

    Returns:
        Power [W].
    """
    pp_forces = case_dir / "postProcessing" / "forces"
    if pp_forces.exists():
        data_files = sorted(pp_forces.glob("**/*"), key=lambda f: f.name)
        if data_files:
            try:
                last_line = data_files[-1].read_text().strip().splitlines()[-1]
                values = [float(v) for v in last_line.replace("(", " ").replace(")", " ").split() if _is_float(v)]
                if len(values) >= 6:
                    # Moment z-component
                    torque_z = abs(values[5])
                    if sizing is not None:
                        vt = sizing.velocity_triangles
                        u2 = vt["outlet"]["u"] if isinstance(vt, dict) else vt.outlet.u
                        omega = u2 / (sizing.impeller_d2 / 2.0)
                        return torque_z * omega
            except (ValueError, IndexError):
                pass

    # Fallback: use sizing estimate
    if sizing is not None:
        return sizing.estimated_power
    return 0.0


def _parse_blade_loading(case_dir: Path) -> list[dict[str, Any]]:
    """Parse blade loading (Cp vs chord) from postProcessing.

    Returns a list of dicts with span, chord, and Cp arrays.
    """
    blade_pp = case_dir / "postProcessing" / "bladeLoading"
    if not blade_pp.exists():
        return []

    results: list[dict[str, Any]] = []
    for span_dir in sorted(blade_pp.iterdir()):
        if span_dir.is_dir():
            try:
                span_frac = float(span_dir.name)
            except ValueError:
                continue
            data_files = sorted(span_dir.glob("*"))
            if data_files:
                try:
                    lines = data_files[-1].read_text().strip().splitlines()
                    chord: list[float] = []
                    cp: list[float] = []
                    for line in lines:
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            chord.append(float(parts[0]))
                            cp.append(float(parts[1]))
                    results.append({
                        "span_fraction": span_frac,
                        "chord": chord,
                        "cp": cp,
                    })
                except (ValueError, OSError):
                    continue
    return results


def _parse_residuals(case_dir: Path) -> list[float]:
    """Parse convergence residuals from the solver log.

    Returns:
        List of final residuals (one per iteration), last entry is
        the most recent.
    """
    residuals: list[float] = []
    for log_name in ["log.simpleFoam", "log", "simpleFoam.log"]:
        log_file = case_dir / log_name
        if log_file.exists():
            try:
                for line in log_file.read_text().splitlines():
                    if "Solving for p" in line and "Final residual" in line:
                        parts = line.split("Final residual = ")
                        if len(parts) > 1:
                            residuals.append(float(parts[1].split(",")[0]))
            except (ValueError, OSError):
                pass
            break
    return residuals


def _is_float(s: str) -> bool:
    """Check if a string can be parsed as float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Geometry adjustment helpers
# ---------------------------------------------------------------------------

def _compute_geometry_adjustment(
    deltas: dict[str, float],
    sizing: SizingResult,
) -> dict[str, float]:
    """Compute geometry corrections based on CFD-vs-target deltas.

    Simple proportional correction:
        - If head is low  → increase beta2 (more backward lean)
        - If head is high → decrease beta2 / trim D2
        - If efficiency is low → reduce blade thickness (not applied here)

    Returns:
        Dictionary of geometry parameter changes (additive).
    """
    changes: dict[str, float] = {}

    rel_head = deltas.get("delta_head_rel", 0.0)
    # Correct beta2 proportionally to head deficit
    # 1% head deficit -> ~0.5 deg beta2 increase
    beta2_correction = -rel_head * 50.0  # degrees per unit relative error
    beta2_correction = max(-3.0, min(3.0, beta2_correction))  # clamp
    changes["delta_beta2_deg"] = beta2_correction

    # D2 trim for large head surplus
    if rel_head > 0.05:
        d2_trim_fraction = -rel_head * 0.5
        d2_trim_fraction = max(-0.05, d2_trim_fraction)
        changes["delta_d2_fraction"] = d2_trim_fraction
    else:
        changes["delta_d2_fraction"] = 0.0

    return changes


def _apply_geometry_changes(
    sizing: SizingResult,
    changes: dict[str, float],
) -> SizingResult:
    """Return a new SizingResult with adjusted geometry parameters.

    This is a shallow mutation — only beta2 and D2 are updated.
    """
    # Apply beta2 correction
    new_beta2 = sizing.beta2 + changes.get("delta_beta2_deg", 0.0)
    new_beta2 = max(15.0, min(45.0, new_beta2))  # physical limits

    # Apply D2 trim
    d2_frac = changes.get("delta_d2_fraction", 0.0)
    new_d2 = sizing.impeller_d2 * (1.0 + d2_frac)

    # Create modified copy (SizingResult is a dataclass)
    import copy
    new_sizing = copy.copy(sizing)
    object.__setattr__(new_sizing, "beta2", new_beta2)
    object.__setattr__(new_sizing, "impeller_d2", new_d2)

    return new_sizing
