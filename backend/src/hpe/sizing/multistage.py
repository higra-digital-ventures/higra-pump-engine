"""Multi-stage turbomachinery sizing.

Distributes total head across multiple stages, sizes each stage
independently, and ensures stage matching (flow continuity,
inter-stage conditions).

Supports:
- Centrifugal pump multi-stage (boiler feed, pipeline)
- Axial compressor multi-stage
- Multi-stage turbines (Francis, axial)

The head split can be equal or optimized per stage based on
specific speed considerations.

References:
    - Gulich (2014). Centrifugal Pumps, Ch. 15 (multi-stage).
    - Japikse (1996). Centrifugal Compressor Design, Ch. 9.
    - Dixon & Hall (2014). Fluid Mechanics & Thermo of Turbomachinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hpe.core.models import G, OperatingPoint, SizingResult
from hpe.sizing.meanline import run_sizing
from hpe.sizing.specific_speed import calc_specific_speed


@dataclass
class StageSpec:
    """Specification for one stage in a multi-stage machine."""

    stage_number: int
    head: float  # Stage head [m]
    flow_rate: float  # Through-flow [m³/s]
    rpm: float  # Rotational speed [rpm]
    nq: float  # Stage specific speed
    inlet_pressure: float = 101325.0  # Inlet stagnation pressure [Pa]
    inlet_temperature: float = 293.15  # Inlet temperature [K]


@dataclass
class StageResult:
    """Result for one stage."""

    stage_number: int
    sizing: SizingResult
    nq: float
    head_fraction: float  # Fraction of total head
    outlet_pressure: float  # [Pa]


@dataclass
class MultiStageResult:
    """Complete multi-stage sizing result."""

    n_stages: int
    total_head: float  # [m]
    total_flow_rate: float  # [m³/s]
    total_power: float  # [W]
    overall_efficiency: float
    stages: list[StageResult]
    warnings: list[str]


def determine_stage_count(
    flow_rate: float,
    total_head: float,
    rpm: float,
    nq_target_min: float = 15.0,
    nq_target_max: float = 60.0,
) -> int:
    """Determine optimal number of stages based on specific speed.

    For centrifugal pumps, Nq between 15-60 gives the best efficiency.
    If the single-stage Nq is too low, we split into multiple stages.

    Args:
        flow_rate: Q [m³/s].
        total_head: Total H [m].
        rpm: Rotational speed [rpm].
        nq_target_min: Minimum desirable Nq.
        nq_target_max: Maximum desirable Nq.

    Returns:
        Recommended number of stages.
    """
    ns_total, nq_total = calc_specific_speed(flow_rate, total_head, rpm)

    if nq_total >= nq_target_min:
        return 1  # Single stage is fine

    # Find n_stages such that nq_per_stage is in target range
    for n in range(2, 20):
        h_per_stage = total_head / n
        _, nq_stage = calc_specific_speed(flow_rate, h_per_stage, rpm)
        if nq_stage >= nq_target_min:
            return n

    return 10  # Maximum practical for centrifugal pumps


def distribute_head(
    total_head: float,
    n_stages: int,
    method: str = "equal",
) -> list[float]:
    """Distribute total head across stages.

    Methods:
        equal: Same head per stage
        optimized: First and last stages slightly lower for better NPSH/matching
        decreasing: Higher head in earlier stages (less cavitation risk)

    Args:
        total_head: Total head [m].
        n_stages: Number of stages.
        method: Distribution method.

    Returns:
        List of head per stage [m].
    """
    if n_stages <= 1:
        return [total_head]

    if method == "equal":
        h = total_head / n_stages
        return [h] * n_stages

    elif method == "optimized":
        # First stage: -10% (better NPSH), last: -5%, rest: compensated
        reduction_first = 0.10
        reduction_last = 0.05
        h_base = total_head / n_stages
        heads = [h_base] * n_stages
        heads[0] = h_base * (1.0 - reduction_first)
        heads[-1] = h_base * (1.0 - reduction_last)
        # Redistribute deficit to middle stages
        deficit = total_head - sum(heads)
        n_mid = max(n_stages - 2, 1)
        for i in range(1, n_stages - 1):
            heads[i] += deficit / n_mid
        return heads

    elif method == "decreasing":
        # Linear decrease: first stage gets most head
        weights = [n_stages - i for i in range(n_stages)]
        total_w = sum(weights)
        return [total_head * w / total_w for w in weights]

    return [total_head / n_stages] * n_stages


def _distribute_head_by_split(
    total_head: float,
    n_stages: int,
    work_split_vector: list[float],
    warnings: list[str],
) -> list[float]:
    """Distribute head using an explicit weight vector.

    The vector is normalised so its elements sum to 1.  If the vector
    is shorter than n_stages the last weight is repeated; if it is
    longer it is truncated.

    Args:
        total_head: Total head [m].
        n_stages: Number of stages.
        work_split_vector: Raw weight values (any positive floats).
        warnings: Mutable list; warning messages are appended here.

    Returns:
        List of head values [m], one per stage, summing to total_head.
    """
    if not work_split_vector or any(w < 0 for w in work_split_vector):
        warnings.append("work_split_vector contains invalid values; falling back to equal split.")
        return [total_head / n_stages] * n_stages

    # Pad or truncate to match n_stages
    vec = list(work_split_vector)
    if len(vec) < n_stages:
        last = vec[-1]
        vec.extend([last] * (n_stages - len(vec)))
    vec = vec[:n_stages]

    total_w = sum(vec)
    if total_w <= 0:
        warnings.append("work_split_vector sums to zero; falling back to equal split.")
        return [total_head / n_stages] * n_stages

    heads = [total_head * w / total_w for w in vec]

    # Warn if any stage head is very small
    for i, h in enumerate(heads):
        if h < 0.05 * total_head:
            warnings.append(
                f"Stage {i+1} work split yields only {h:.1f} m "
                f"({h/total_head*100:.1f}% of total). Check split vector."
            )

    return heads


# ---------------------------------------------------------------------------
# Interstage loss model
# ---------------------------------------------------------------------------

@dataclass
class InterstageModel:
    """Losses in the return channel / crossover between stages.

    References:
        Gulich (2014) Ch. 15 — return channel and seal leakage losses.
    """

    return_channel_loss_fraction: float = 0.03  # 2-5% of stage head
    seal_leakage_fraction: float = 0.015  # 1-2% of flow rate
    disc_friction_fraction: float = 0.01  # ~1% of stage power

    def head_loss(self, stage_head: float) -> float:
        """Return channel head loss [m]."""
        return stage_head * self.return_channel_loss_fraction

    def leakage_flow(self, flow_rate: float) -> float:
        """Seal leakage flow [m3/s]."""
        return flow_rate * self.seal_leakage_fraction

    def disc_friction_power(self, stage_power: float) -> float:
        """Disc friction power loss [W]."""
        return stage_power * self.disc_friction_fraction


@dataclass
class StageDetail:
    """Extended per-stage result with interstage losses and thermodynamics."""

    stage_number: int
    head: float  # Gross head [m]
    net_head: float  # Head after interstage losses [m]
    efficiency: float  # Stage hydraulic efficiency [-]
    d2: float  # Outlet diameter [m]
    nq: float  # Specific speed
    power: float  # Stage power [W]
    interstage_head_loss: float  # Return channel loss [m]
    seal_leakage_flow: float  # Leakage [m3/s]
    disc_friction_power: float  # Disc friction [W]
    inlet_pressure: float  # Stage inlet pressure [Pa]
    outlet_pressure: float  # Stage outlet pressure [Pa]
    temperature_rise: float  # Fluid temperature rise [K]


@dataclass
class MultiStageDesignerResult:
    """Enhanced multi-stage result with thermodynamic detail."""

    n_stages: int
    total_head: float  # [m]
    total_flow_rate: float  # [m3/s]
    total_power: float  # [W]
    overall_efficiency: float
    mechanical_efficiency: float
    seal_efficiency: float
    stages: list[StageDetail]
    stage_count_optimization: list[dict]  # [{n_stages, eta_total}]
    warnings: list[str]


class MultiStageDesigner:
    """Advanced multi-stage pump design with thermodynamic completeness.

    Features:
        - Auto stage count selection from Nq correlation
        - Interstage loss model (return channel, seals, disc friction)
        - Per-stage thermodynamic tracking (temperature rise)
        - Stage count vs efficiency trade-off analysis
        - Total efficiency: eta_total = product(eta_stage_i) * eta_mech * eta_seal

    Args:
        total_head: Total required head [m].
        flow_rate: Design flow rate [m3/s].
        rpm: Rotational speed [rpm].
        n_stages: Number of stages (None for auto).
        rho: Fluid density [kg/m3].
        cp: Fluid specific heat [J/(kg.K)].
        p_inlet: Inlet pressure [Pa].
        t_inlet: Inlet temperature [K].
        eta_mechanical: Mechanical efficiency [-].
        interstage: Interstage loss model parameters.
    """

    def __init__(
        self,
        total_head: float,
        flow_rate: float,
        rpm: float,
        n_stages: int | None = None,
        rho: float = 998.2,
        cp: float = 4182.0,
        p_inlet: float = 101325.0,
        t_inlet: float = 293.15,
        eta_mechanical: float = 0.96,
        interstage: InterstageModel | None = None,
    ) -> None:
        self.total_head = total_head
        self.flow_rate = flow_rate
        self.rpm = rpm
        self.rho = rho
        self.cp = cp
        self.p_inlet = p_inlet
        self.t_inlet = t_inlet
        self.eta_mechanical = eta_mechanical
        self.interstage = interstage or InterstageModel()

        self._n_stages = n_stages
        self._warnings: list[str] = []

    def _auto_stage_count(self) -> int:
        """Determine optimal stage count from Nq correlation.

        Uses: n = ceil(total_head / H_per_stage_max)
        where H_per_stage_max comes from Nq_target = 25-40 range.
        """
        # Target Nq for best efficiency: 25-40 for centrifugal pumps
        nq_target = 30.0

        # From Nq = n * Q^0.5 / H^0.75, solve for H_per_stage
        # Nq = rpm * Q^0.5 / (g * H_stage)^0.75
        # H_stage = (rpm * Q^0.5 / Nq)^(4/3) / g
        q_sqrt = self.flow_rate ** 0.5
        h_per_stage_max = ((self.rpm * q_sqrt / nq_target) ** (4.0 / 3.0)) / G

        if h_per_stage_max <= 0:
            return 1

        n = math.ceil(self.total_head / h_per_stage_max)
        return max(1, min(n, 15))

    def _compute_stage_efficiency(self, nq: float) -> float:
        """Estimate stage hydraulic efficiency from Nq.

        Gulich correlation for centrifugal pumps.
        """
        if nq < 10:
            return 0.70
        elif nq < 20:
            return 0.75 + 0.005 * (nq - 10)
        elif nq < 40:
            return 0.80 + 0.003 * (nq - 20)
        elif nq < 70:
            return 0.86 + 0.001 * (nq - 40)
        else:
            return 0.89

    def design(self) -> MultiStageDesignerResult:
        """Run the full multi-stage design.

        Returns:
            :class:`MultiStageDesignerResult` with per-stage details
            and overall performance.
        """
        self._warnings = []

        # Stage count
        n_stages = self._n_stages if self._n_stages is not None else self._auto_stage_count()

        if n_stages < 1:
            n_stages = 1
            self._warnings.append("Stage count forced to 1.")

        # Head distribution (equal, with interstage losses accounted for)
        gross_head_per_stage = self.total_head / n_stages

        stages: list[StageDetail] = []
        p_current = self.p_inlet
        t_current = self.t_inlet
        total_power = 0.0
        product_eta = 1.0

        for i in range(n_stages):
            # Per-stage sizing
            _, nq = calc_specific_speed(self.flow_rate, gross_head_per_stage, self.rpm)

            op = OperatingPoint(
                flow_rate=self.flow_rate,
                head=gross_head_per_stage,
                rpm=self.rpm,
            )
            sizing = run_sizing(op)

            # Interstage losses (applied to all but last stage)
            if i < n_stages - 1:
                hl = self.interstage.head_loss(gross_head_per_stage)
                leak = self.interstage.leakage_flow(self.flow_rate)
                disc = self.interstage.disc_friction_power(sizing.estimated_power)
            else:
                hl = 0.0
                leak = 0.0
                disc = 0.0

            net_head = gross_head_per_stage - hl
            stage_power = sizing.estimated_power + disc

            # Pressure at outlet
            p_out = p_current + self.rho * G * net_head

            # Temperature rise: dT = P_loss / (m_dot * cp)
            p_loss = stage_power - self.rho * G * self.flow_rate * net_head
            dt = p_loss / (self.rho * self.flow_rate * self.cp) if (self.rho * self.flow_rate * self.cp) > 0 else 0.0
            dt = max(0.0, dt)

            t_current += dt

            stage_eta = sizing.estimated_efficiency
            product_eta *= stage_eta

            stages.append(StageDetail(
                stage_number=i + 1,
                head=gross_head_per_stage,
                net_head=net_head,
                efficiency=stage_eta,
                d2=sizing.impeller_d2,
                nq=nq,
                power=stage_power,
                interstage_head_loss=hl,
                seal_leakage_flow=leak,
                disc_friction_power=disc,
                inlet_pressure=p_current,
                outlet_pressure=p_out,
                temperature_rise=dt,
            ))

            total_power += stage_power
            p_current = p_out

            # Warnings
            if nq < 10:
                self._warnings.append(f"Stage {i+1}: Nq={nq:.1f} is very low.")
            if nq > 80:
                self._warnings.append(f"Stage {i+1}: Nq={nq:.1f} is too high for centrifugal.")
            if dt > 2.0:
                self._warnings.append(f"Stage {i+1}: Temperature rise {dt:.1f} K is significant.")

        # Seal efficiency: accounts for leakage
        total_leakage = sum(s.seal_leakage_flow for s in stages)
        eta_seal = (self.flow_rate - total_leakage) / self.flow_rate if self.flow_rate > 0 else 1.0

        # Overall efficiency
        overall_eta = product_eta * self.eta_mechanical * eta_seal

        # Stage count optimization analysis
        stage_count_opt = self._optimize_stage_count()

        return MultiStageDesignerResult(
            n_stages=n_stages,
            total_head=self.total_head,
            total_flow_rate=self.flow_rate,
            total_power=total_power,
            overall_efficiency=overall_eta,
            mechanical_efficiency=self.eta_mechanical,
            seal_efficiency=eta_seal,
            stages=stages,
            stage_count_optimization=stage_count_opt,
            warnings=self._warnings,
        )

    def _optimize_stage_count(self) -> list[dict]:
        """Evaluate efficiency vs stage count trade-off.

        Returns list of {n_stages, eta_total, nq_per_stage} for
        n=1..max_reasonable.
        """
        results: list[dict] = []
        max_n = min(12, max(2, self._auto_stage_count() * 2))

        for n in range(1, max_n + 1):
            h_per = self.total_head / n
            _, nq = calc_specific_speed(self.flow_rate, h_per, self.rpm)
            stage_eta = self._compute_stage_efficiency(nq)

            # Interstage losses reduce overall efficiency
            n_interstage = max(0, n - 1)
            loss_frac = 1.0 - n_interstage * self.interstage.return_channel_loss_fraction
            eta_total = (stage_eta ** n) * self.eta_mechanical * loss_frac

            results.append({
                "n_stages": n,
                "eta_total": round(eta_total, 4),
                "nq_per_stage": round(nq, 1),
                "head_per_stage": round(h_per, 2),
            })

        return results


def size_multistage(
    flow_rate: float,
    total_head: float,
    rpm: float,
    n_stages: int | None = None,
    head_distribution: str = "equal",
    rho: float = 998.2,
    p_inlet: float = 101325.0,
    work_split_vector: list[float] | None = None,
) -> MultiStageResult:
    """Size a complete multi-stage machine.

    Args:
        flow_rate: Q [m³/s].
        total_head: Total H [m].
        rpm: Rotational speed [rpm].
        n_stages: Number of stages. If None, auto-determined.
        head_distribution: Head split method (equal/optimized/decreasing).
            Ignored when work_split_vector is provided.
        rho: Fluid density [kg/m³].
        p_inlet: Inlet pressure [Pa].
        work_split_vector: Optional list of weights (one per stage) for
            distributing total head unevenly. Will be normalised to sum to 1.
            E.g. [0.4, 0.3, 0.3] gives the first stage 40% of total head.
            Overrides head_distribution when provided.

    Returns:
        MultiStageResult with all stage details.
    """
    warnings: list[str] = []

    if n_stages is None:
        n_stages = determine_stage_count(flow_rate, total_head, rpm)

    # work_split_vector takes precedence over head_distribution method
    if work_split_vector is not None:
        heads = _distribute_head_by_split(total_head, n_stages, work_split_vector, warnings)
    else:
        heads = distribute_head(total_head, n_stages, head_distribution)

    stages: list[StageResult] = []
    total_power = 0.0
    p_current = p_inlet

    for i in range(n_stages):
        h_stage = heads[i]
        _, nq = calc_specific_speed(flow_rate, h_stage, rpm)

        op = OperatingPoint(
            flow_rate=flow_rate,
            head=h_stage,
            rpm=rpm,
        )
        sizing = run_sizing(op)

        # Outlet pressure of this stage = inlet + rho*g*H
        p_out = p_current + rho * G * h_stage

        stages.append(StageResult(
            stage_number=i + 1,
            sizing=sizing,
            nq=nq,
            head_fraction=h_stage / total_head,
            outlet_pressure=p_out,
        ))

        total_power += sizing.estimated_power
        p_current = p_out

        # Warnings
        if nq < 10:
            warnings.append(f"Stage {i+1}: Nq={nq:.1f} is very low. Risk of poor efficiency.")
        if nq > 80:
            warnings.append(f"Stage {i+1}: Nq={nq:.1f} is high. Consider mixed-flow design.")

    # Overall efficiency
    useful_power = rho * G * flow_rate * total_head
    overall_eta = useful_power / total_power if total_power > 0 else 0.0

    # Check diameter consistency (all stages should be similar for common shaft)
    d2_values = [s.sizing.impeller_d2 for s in stages]
    if len(d2_values) > 1:
        d2_spread = (max(d2_values) - min(d2_values)) / max(d2_values)
        if d2_spread > 0.15:
            warnings.append(
                f"D2 varies {d2_spread*100:.0f}% across stages. "
                "May need impeller trimming or different head split."
            )

    return MultiStageResult(
        n_stages=n_stages,
        total_head=total_head,
        total_flow_rate=flow_rate,
        total_power=total_power,
        overall_efficiency=overall_eta,
        stages=stages,
        warnings=warnings,
    )
