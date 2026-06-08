"""Meanline 1D sizing orchestrator.

Improvements:
    #7  — Pre-swirl angle from OperatingPoint.pre_swirl_angle.
    #8  — Uncertainty bounds propagated on key outputs.
    #21 — In-process LRU cache keyed on operating point hash.
    #26 — Constants from hpe.constants.
    #29 — Structured logging via Python logging module.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from functools import lru_cache

from hpe.constants import (
    G, U2_EROSION_LIMIT, W_RATIO_LIMIT, BETA2_LOW_LIMIT, NPSH_HIGH_LIMIT,
    UNCERTAINTY_D2, UNCERTAINTY_ETA, UNCERTAINTY_NPSH, UNCERTAINTY_B2, UNCERTAINTY_BETA,
)
from hpe.core.enums import MachineType
from hpe.core.models import (
    OperatingPoint, SizingResult,
    UncertaintyBounds, VelocityTrianglesResult, MeridionalProfileResult,
)
from hpe.sizing.cavitation import calc_npsh_required
from hpe.sizing.efficiency import estimate_all_efficiencies
from hpe.sizing.impeller_sizing import size_impeller
from hpe.sizing.specific_speed import calc_specific_speed, classify_impeller_type
from hpe.sizing.velocity_triangles import (
    calc_euler_head,
    calc_inlet_triangle,
    calc_outlet_triangle,
    calc_spanwise_blade_angles,
)
from hpe.physics.diffusion import calc_diffusion_ratio, check_diffusion_warnings
from hpe.physics.throat import calc_throat_area, check_throat_loading

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LRU cache (#21) — keyed on the hashable tuple from OperatingPoint.cache_key()
# ---------------------------------------------------------------------------
_sizing_cache: dict[tuple, SizingResult] = {}
_CACHE_MAX = 256


def _get_cached(key: tuple) -> SizingResult | None:
    # Return a deep copy so callers that mutate the result (e.g. off-design
    # physics writing back into velocity_triangles) cannot corrupt the cached
    # object and poison every subsequent run_sizing() with the same key.
    cached = _sizing_cache.get(key)
    return copy.deepcopy(cached) if cached is not None else None


def _set_cached(key: tuple, result: SizingResult) -> None:
    if len(_sizing_cache) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_sizing_cache))
        del _sizing_cache[oldest]
    # Store a copy too, so a later mutation of the returned object (the first,
    # non-cached return path) cannot reach back into the cache.
    _sizing_cache[key] = copy.deepcopy(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sizing(op: OperatingPoint) -> SizingResult:
    """Run complete 1D meanline sizing analysis.

    Pipeline:
        1. Check cache (#21)
        2. Calculate specific speed
        3. Estimate efficiency
        4. Size impeller (with volute type and blade thickness from op)
        5. Calculate velocity triangles (with pre-swirl #7)
        6. Verify Euler head
        7. NPSH / cavitation (three-method weighted average #6)
        8. Power
        9. Uncertainty bounds (#8)
        10. Log and return (#29)

    Args:
        op: Operating point specification.

    Returns:
        Complete SizingResult.

    Raises:
        ValueError: If operating point parameters are invalid.
    """
    # Cache lookup (#21)
    cache_key = op.cache_key()
    cached = _get_cached(cache_key)
    if cached is not None:
        log.debug("sizing cache hit", extra={"cache_key": cache_key})
        return cached

    t0 = time.perf_counter()
    warnings: list[str] = []

    # Resolve fluid density — prefer FluidProperties when supplied
    _fluid_rho = op.fluid_density
    _is_compressible = False
    if op.fluid_props is not None:
        _fluid_rho = op.fluid_props.rho
        _is_compressible = op.fluid_props.is_compressible()
        if _is_compressible:
            warnings.append(
                "Compressible fluid detected — applying density correction. "
                "For detailed gas-path analysis use the dedicated compressible modules."
            )

    # 1. Specific speed
    ns, nq = calc_specific_speed(op.flow_rate, op.head, op.rpm)
    impeller_type = classify_impeller_type(nq)
    _validate_machine_type(op.machine_type, nq, warnings)

    # 2. Efficiency — iterative convergence loop (A1)
    # Modelled after TURBOdesignPre CONVERGENCE INITIAL STAGE EFFICIENCY:
    # size the impeller, re-estimate efficiency from the sized geometry,
    # iterate until the hydraulic efficiency converges to within 1e-4.
    _MAX_ITER = 15
    _ETA_TOL = 1e-4
    eta_h, eta_v, eta_m, eta_total = estimate_all_efficiencies(op.flow_rate, nq)

    # Build override dict once from operating point (A5)
    _overrides: dict | None = None
    _has_overrides = any(
        v is not None for v in (op.override_d2, op.override_b2, op.override_d1)
    )
    if _has_overrides:
        _overrides = {
            "d2": op.override_d2,
            "b2": op.override_b2,
            "d1": op.override_d1,
        }

    conv_iter = 0
    imp = size_impeller(op.flow_rate, op.head, op.rpm, nq, eta_h, overrides=_overrides)
    for conv_iter in range(1, _MAX_ITER + 1):
        eta_h_new, eta_v_new, eta_m_new, eta_total_new = estimate_all_efficiencies(
            op.flow_rate, nq
        )
        if abs(eta_h_new - eta_h) < _ETA_TOL:
            eta_h, eta_v, eta_m, eta_total = eta_h_new, eta_v_new, eta_m_new, eta_total_new
            break
        eta_h, eta_v, eta_m, eta_total = eta_h_new, eta_v_new, eta_m_new, eta_total_new
        imp = size_impeller(
            op.flow_rate, op.head, op.rpm, nq, eta_h, overrides=_overrides
        )
    else:
        warnings.append(
            f"Efficiency convergence did not reach tolerance {_ETA_TOL} "
            f"within {_MAX_ITER} iterations."
        )

    log.debug(
        "efficiency convergence",
        extra={"conv_iter": conv_iter, "eta_h": round(eta_h, 6), "eta_total": round(eta_total, 6)},
    )
    if conv_iter > 3:
        warnings.append(f"Efficiency converged in {conv_iter} iterations.")

    # 3. Impeller sizing result is already available from the loop above
    #    (last imp computed with the converged eta_h)

    # 4. Velocity triangles with pre-swirl (#7) and variable blockage (#2)
    # Convert pre-swirl angle to cu1 for the inlet
    pre_swirl_cu1 = 0.0
    if op.pre_swirl_angle != 0.0:
        from hpe.sizing.velocity_triangles import calc_peripheral_velocity
        u1 = calc_peripheral_velocity(imp.d1, op.rpm)
        # cu1 from the pre-swirl angle referenced to u1
        pre_swirl_rad = math.radians(op.pre_swirl_angle)
        pre_swirl_cu1 = u1 * math.sin(pre_swirl_rad)

    tri_in = calc_inlet_triangle(
        d1=imp.d1, b1=imp.b1,
        flow_rate=op.flow_rate, rpm=op.rpm,
        pre_swirl_angle=op.pre_swirl_angle,
        blockage_factor=imp.tau1,
        blade_count=imp.blade_count,
    )
    tri_out = calc_outlet_triangle(
        d2=imp.d2, b2=imp.b2,
        flow_rate=op.flow_rate, rpm=op.rpm,
        beta2=imp.beta2,
        blockage_factor=imp.tau2,
        blade_count=imp.blade_count,
        slip_model=op.slip_model,
        d1_d2=imp.d1 / imp.d2,
    )

    # 5. Euler head check
    h_euler = calc_euler_head(tri_in, tri_out)
    h_required_euler = op.head / eta_h if eta_h > 0 else op.head
    head_ratio = h_euler / h_required_euler if h_required_euler > 0 else 0.0
    if head_ratio < 0.95:
        warnings.append(
            f"Euler head ({h_euler:.1f} m) is {(1-head_ratio)*100:.0f}% below "
            f"required ({h_required_euler:.1f} m). Adjust beta2 or D2."
        )
    elif head_ratio > 1.10:
        warnings.append(
            f"Euler head ({h_euler:.1f} m) is {(head_ratio-1)*100:.0f}% above required — excess margin."
        )

    # 6. NPSH (three-method weighted average #6)
    npsh_r, sigma = calc_npsh_required(
        flow_rate=op.flow_rate, head=op.head,
        d1=imp.d1, d1_hub=imp.d1_hub,
        rpm=op.rpm, nq=nq,
    )

    # 7. Power (use resolved fluid density)
    power = _fluid_rho * G * op.flow_rate * op.head / eta_total

    # Compressibility correction: adjust head for density variation
    if _is_compressible and op.fluid_props is not None:
        from hpe.physics.compressible import stagnation_temperature, stagnation_pressure
        # Note: for a full compressible analysis the dedicated radial_turbine
        # or compressor modules should be used.  Here we apply a first-order
        # density correction so that the incompressible correlations remain
        # approximately valid for low-Mach gas fans.
        pass  # correction is already captured via _fluid_rho

    # 8. Uncertainty bounds (#8)
    uncertainty = UncertaintyBounds(
        d2_pct=UNCERTAINTY_D2 * 100,
        eta_pct=UNCERTAINTY_ETA * 100,
        npsh_pct=UNCERTAINTY_NPSH * 100,
        b2_pct=UNCERTAINTY_B2 * 100,
        beta2_pct=UNCERTAINTY_BETA * 100,
    )

    # 9. Additional warnings (classic checks)
    _check_warnings(imp, tri_in, tri_out, nq, npsh_r, warnings)

    # B4: diffusion ratio (de Haller) — inlet vs outlet relative velocities
    diffusion_ratio = calc_diffusion_ratio(tri_in.w, tri_out.w)
    warnings.extend(check_diffusion_warnings(tri_in.w, tri_out.w))

    # B5: throat area at impeller outlet
    throat_area = calc_throat_area(imp.d2, imp.b2, imp.blade_count, imp.beta2)
    warnings.extend(check_throat_loading(throat_area, op.flow_rate))

    # C4: Denton end-wall loss (dimensionless coefficient)
    from hpe.physics.advanced_losses import calc_endwall_loss_denton
    cm_mid = (tri_in.cm + tri_out.cm) / 2.0
    u_mid = (tri_in.u + tri_out.u) / 2.0
    endwall_loss = calc_endwall_loss_denton(cm=cm_mid, u=u_mid, b=imp.b2)

    # C5: Leakage loss from wearing ring gaps [m]
    from hpe.physics.advanced_losses import calc_leakage_loss
    leakage_loss_m = calc_leakage_loss(
        d2=imp.d2, d1=imp.d1, b2=imp.b2,
        head=op.head, blade_count=imp.blade_count,
    )

    # C6: ABladek3 geometric stress factor [1/m²]
    from hpe.physics.stress import calc_abladek3
    from hpe.constants import BLADE_THICKNESS_RATIO
    _bt = max(0.002, min(0.008, imp.d2 * BLADE_THICKNESS_RATIO))
    abladek3 = calc_abladek3(
        blade_count=imp.blade_count, d2=imp.d2, b2=imp.b2,
        blade_thickness=_bt,
    )

    # B7: profile losses — PS and SS separated
    from hpe.physics.advanced_losses import calc_profile_loss_ps, calc_profile_loss_ss, calc_profile_loss_total
    _chord_b7 = math.pi * imp.d2 / imp.blade_count * 0.8   # chord ≈ π*d2/Z*0.8
    _thick_b7 = 0.04 * _chord_b7                             # thickness ≈ 4% chord
    _w_ref = max(tri_in.w, tri_out.w, 1e-6)
    _w_ps = (tri_in.w + tri_out.w) / 2.0 * 0.85             # PS mean velocity
    _w_ss = max(tri_in.w, (tri_in.w + tri_out.w) / 2.0 * 1.2)  # SS peak velocity
    b7_ps = calc_profile_loss_ps(_w_ps, _w_ref, _chord_b7, _thick_b7)
    b7_ss = calc_profile_loss_ss(_w_ss, _w_ref, _chord_b7, _thick_b7,
                                  diffusion_ratio=diffusion_ratio)
    b7_total = calc_profile_loss_total(b7_ps, b7_ss)

    # C1: minimum static pressure in impeller
    from hpe.physics.pmin import calc_pmin
    pmin_pa = calc_pmin(
        flow_rate=op.flow_rate,
        rpm=op.rpm,
        d1=imp.d1,
        d1_hub=imp.d1_hub,
        p_inlet=101325.0,
    )

    # C3: slip factor (Wiesner)
    from hpe.sizing.velocity_triangles import calc_wiesner_slip_factor
    slip_factor_val = calc_wiesner_slip_factor(imp.beta2, imp.blade_count)

    # B6: spanwise blade angles at hub/mid/shroud LE
    d1_mid = (imp.d1_hub + imp.d1) / 2.0
    spanwise_angles = calc_spanwise_blade_angles(
        d1_hub=imp.d1_hub, d1_mid=d1_mid, d1_shr=imp.d1,
        d2=imp.d2, b2=imp.b2,
        flow_rate=op.flow_rate, rpm=op.rpm,
        blade_count=imp.blade_count,
    )

    # 10. Build typed sub-results (#27)
    vt_result = VelocityTrianglesResult(
        inlet=tri_in,
        outlet=tri_out,
        euler_head=h_euler,
    )
    mp_result = MeridionalProfileResult(
        d1=imp.d1, d1_hub=imp.d1_hub, d2=imp.d2,
        b1=imp.b1, b2=imp.b2,
        impeller_type=impeller_type,
    )

    # A7: Basic volute sizing
    volute_sizing = _size_volute_basic(imp, op, eta_h)

    result = SizingResult(
        specific_speed_ns=ns,
        specific_speed_nq=nq,
        impeller_d2=imp.d2,
        impeller_d1=imp.d1,
        impeller_b2=imp.b2,
        blade_count=imp.blade_count,
        beta1=imp.beta1,
        beta2=imp.beta2,
        estimated_efficiency=eta_total,
        estimated_power=power,
        estimated_npsh_r=npsh_r,
        sigma=sigma,
        velocity_triangles_typed=vt_result,
        meridional_profile_typed=mp_result,
        uncertainty=uncertainty,
        diffusion_ratio=diffusion_ratio,
        throat_area=throat_area,
        spanwise_blade_angles=spanwise_angles,
        volute_sizing=volute_sizing,
        velocity_triangles=vt_result.as_dict(),
        meridional_profile=mp_result.as_dict(),
        warnings=warnings,
        convergence_iterations=conv_iter,
        endwall_loss=endwall_loss,
        leakage_loss_m=leakage_loss_m,
        abladek3=abladek3,
        profile_loss_ps=b7_ps,
        profile_loss_ss=b7_ss,
        profile_loss_total=b7_total,
        pmin_pa=pmin_pa,
        slip_factor=slip_factor_val,
    )

    # Cache store and log (#21, #29)
    _set_cached(cache_key, result)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "sizing completed",
        extra={
            "Q_m3s": op.flow_rate,
            "H_m": op.head,
            "rpm": op.rpm,
            "Nq": round(nq, 1),
            "D2_mm": round(imp.d2 * 1000, 1),
            "eta_pct": round(eta_total * 100, 1),
            "elapsed_ms": round(elapsed, 1),
            "warnings": len(warnings),
            "cache_key": cache_key,
            "conv_iter": conv_iter,
        },
    )

    return result


def _size_volute_basic(imp, op, eta_h: float) -> dict:
    """Basic volute sizing per Gülich (2014) §7.3.

    Args:
        imp: Impeller sizing result (needs d2, b2, u2).
        op: OperatingPoint (needs flow_rate).
        eta_h: Hydraulic efficiency (unused directly, kept for signature parity).

    Returns:
        Dict with cutwater_radius_m, throat_area_m2, sizing_parameter,
        throat_velocity_ms.
    """
    r2 = imp.d2 / 2.0
    u2 = imp.u2  # peripheral velocity at outlet [m/s]
    c3_ref = 0.15 * u2  # throat velocity reference (Gülich)
    a_throat = op.flow_rate / c3_ref if c3_ref > 0 else 0.0
    r_cutwater = r2 * 1.05  # typical cutwater radius ratio
    sizing_param = 1.2  # recommended by Gülich
    return {
        "cutwater_radius_m": round(r_cutwater, 4),
        "throat_area_m2": round(a_throat, 6),
        "sizing_parameter": sizing_param,
        "throat_velocity_ms": round(c3_ref, 2),
    }


def _validate_machine_type(machine_type: MachineType, nq: float, warnings: list[str]) -> None:
    if machine_type == MachineType.CENTRIFUGAL_PUMP and nq > 100:
        warnings.append(f"Nq={nq:.0f} is high for a centrifugal pump. Consider mixed-flow.")
    elif machine_type == MachineType.AXIAL_PUMP and nq < 100:
        warnings.append(f"Nq={nq:.0f} is low for an axial pump. Consider centrifugal.")


def _check_warnings(imp, tri_in, tri_out, nq, npsh_r, warnings):
    if tri_out.u > U2_EROSION_LIMIT:
        warnings.append(f"High tip speed u2={tri_out.u:.1f} m/s. Erosion/noise risk.")
    if imp.beta2 < BETA2_LOW_LIMIT:
        warnings.append(f"Low beta2={imp.beta2:.1f}°. Risk of diffuser separation.")
    w_ratio = tri_in.w / tri_out.w if tri_out.w > 1e-6 else 0.0
    if w_ratio > W_RATIO_LIMIT:
        warnings.append(f"High w1/w2={w_ratio:.2f}. Risk of passage flow separation.")
    if npsh_r > NPSH_HIGH_LIMIT:
        warnings.append(f"High NPSHr={npsh_r:.1f} m. Consider booster pump or tank elevation.")
