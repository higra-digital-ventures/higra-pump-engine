"""Preliminary design database for turbomachinery types.

Provides empirical correlations (Gülich, Stepanoff, Pfleiderer, Cordier diagram)
for quick preliminary design — equivalent to the TURBOdesignPre database.

Supported machine types:
    centrifugal_pump        — standard radial impeller
    mixed_flow_pump         — mixed-flow/diagonal impeller
    axial_pump              — axial fan/pump
    francis_turbine         — Francis runner (pump-turbine)
    centrifugal_compressor  — radial compressor
    axial_compressor        — axial compressor stage
    sirocco_fan             — forward-curved sirocco
    return_channel          — vaned diffuser / return channel
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DesignDBEntry:
    """Preliminary design recommendations for a machine type."""
    machine_type: str
    # Specific speed ranges (Nq = N·√Q / H^0.75 in SI)
    nq_min: float
    nq_max: float
    nq_optimal: float           # Best efficiency Nq
    # Geometry correlations (Gülich / Stepanoff)
    d2_nq_exponent: float       # D2 ∝ Nq^(-exponent)
    b2_d2_ratio_ref: float      # b2/D2 at reference Nq
    b2_d2_nq_slope: float       # d(b2/D2)/d(Nq) slope
    blade_count_min: int
    blade_count_max: int
    blade_count_formula: str    # 'gulich' | 'pfleiderer' | 'fixed'
    beta2_min_deg: float
    beta2_max_deg: float
    beta2_ref_deg: float        # typical β2 at optimal Nq
    # Efficiency
    eta_ref: float              # reference efficiency at Nq_opt
    eta_nq_slope: float         # d(eta)/d(Nq) — efficiency change per Nq unit
    # Flow coefficient φ = Cm2/(u2)
    phi_min: float
    phi_max: float
    phi_ref: float
    # Head coefficient ψ = gH/u2²
    psi_min: float
    psi_max: float
    psi_ref: float
    # Notes
    notes: str = ''
    warnings_at_limits: list[str] = field(default_factory=list)


# Machine type database (empirical correlations from Gülich §3, Stepanoff, Cordier)
DESIGN_DATABASE: dict[str, DesignDBEntry] = {
    'centrifugal_pump': DesignDBEntry(
        machine_type='centrifugal_pump',
        nq_min=5, nq_max=80, nq_optimal=30,
        d2_nq_exponent=0.50,
        b2_d2_ratio_ref=0.065, b2_d2_nq_slope=0.0008,
        blade_count_min=5, blade_count_max=9, blade_count_formula='gulich',
        beta2_min_deg=17, beta2_max_deg=35, beta2_ref_deg=22,
        eta_ref=0.85, eta_nq_slope=0.0008,
        phi_min=0.02, phi_max=0.15, phi_ref=0.07,
        psi_min=0.35, psi_max=0.70, psi_ref=0.50,
        notes='Standard radial centrifugal pump. Optimal Nq 25-35.',
        warnings_at_limits=['Nq < 10: very low, consider reducing rpm', 'Nq > 70: mixed-flow more efficient'],
    ),
    'mixed_flow_pump': DesignDBEntry(
        machine_type='mixed_flow_pump',
        nq_min=50, nq_max=160, nq_optimal=90,
        d2_nq_exponent=0.65,
        b2_d2_ratio_ref=0.18, b2_d2_nq_slope=0.0015,
        blade_count_min=4, blade_count_max=7, blade_count_formula='gulich',
        beta2_min_deg=20, beta2_max_deg=50, beta2_ref_deg=35,
        eta_ref=0.87, eta_nq_slope=0.0005,
        phi_min=0.10, phi_max=0.30, phi_ref=0.18,
        psi_min=0.25, psi_max=0.55, psi_ref=0.38,
        notes='Diagonal/mixed-flow pump. Nq 60-130.',
        warnings_at_limits=['Nq < 50: use centrifugal', 'Nq > 150: use axial'],
    ),
    'axial_pump': DesignDBEntry(
        machine_type='axial_pump',
        nq_min=100, nq_max=400, nq_optimal=200,
        d2_nq_exponent=0.80,
        b2_d2_ratio_ref=0.50, b2_d2_nq_slope=0.002,
        blade_count_min=3, blade_count_max=6, blade_count_formula='fixed',
        beta2_min_deg=10, beta2_max_deg=30, beta2_ref_deg=18,
        eta_ref=0.88, eta_nq_slope=0.00025,
        phi_min=0.20, phi_max=0.50, phi_ref=0.32,
        psi_min=0.10, psi_max=0.35, psi_ref=0.20,
        notes='Axial pump/propeller. Nq > 120.',
        warnings_at_limits=['Nq < 100: mixed-flow more efficient', 'Nq > 350: propeller'],
    ),
    'francis_turbine': DesignDBEntry(
        machine_type='francis_turbine',
        nq_min=50, nq_max=300, nq_optimal=140,
        d2_nq_exponent=0.50,
        b2_d2_ratio_ref=0.25, b2_d2_nq_slope=0.001,
        blade_count_min=9, blade_count_max=19, blade_count_formula='fixed',
        beta2_min_deg=15, beta2_max_deg=45, beta2_ref_deg=28,
        eta_ref=0.92, eta_nq_slope=0.0003,
        phi_min=0.08, phi_max=0.40, phi_ref=0.22,
        psi_min=0.30, psi_max=0.90, psi_ref=0.55,
        notes='Francis turbine runner. Nq 60-300 (Gülich §17).',
        warnings_at_limits=['Nq < 60: Pelton turbine range', 'Nq > 280: Kaplan preferred'],
    ),
    'centrifugal_compressor': DesignDBEntry(
        machine_type='centrifugal_compressor',
        nq_min=10, nq_max=100, nq_optimal=40,
        d2_nq_exponent=0.50,
        b2_d2_ratio_ref=0.07, b2_d2_nq_slope=0.0009,
        blade_count_min=12, blade_count_max=22, blade_count_formula='pfleiderer',
        beta2_min_deg=20, beta2_max_deg=55, beta2_ref_deg=40,
        eta_ref=0.80, eta_nq_slope=0.0005,
        phi_min=0.01, phi_max=0.20, phi_ref=0.08,
        psi_min=0.40, psi_max=0.80, psi_ref=0.58,
        notes='Radial compressor. u2 up to 450 m/s. Splitters common at Nq>40.',
        warnings_at_limits=['u2 > 350 m/s: check compressibility', 'Consider splitter blades above Nq=40'],
    ),
    'axial_compressor': DesignDBEntry(
        machine_type='axial_compressor',
        nq_min=100, nq_max=500, nq_optimal=250,
        d2_nq_exponent=0.75,
        b2_d2_ratio_ref=0.20, b2_d2_nq_slope=0.001,
        blade_count_min=14, blade_count_max=32, blade_count_formula='fixed',
        beta2_min_deg=15, beta2_max_deg=45, beta2_ref_deg=30,
        eta_ref=0.88, eta_nq_slope=0.0002,
        phi_min=0.30, phi_max=0.70, phi_ref=0.50,
        psi_min=0.20, psi_max=0.60, psi_ref=0.35,
        notes='Axial compressor stage. De Haller > 0.72 required.',
        warnings_at_limits=['De Haller < 0.72: risk of stall', 'Work coefficient > 0.55: high loading'],
    ),
    'sirocco_fan': DesignDBEntry(
        machine_type='sirocco_fan',
        nq_min=150, nq_max=600, nq_optimal=300,
        d2_nq_exponent=0.70,
        b2_d2_ratio_ref=0.40, b2_d2_nq_slope=0.001,
        blade_count_min=24, blade_count_max=64, blade_count_formula='fixed',
        beta2_min_deg=90, beta2_max_deg=135, beta2_ref_deg=110,  # forward-curved
        eta_ref=0.65, eta_nq_slope=0.0001,
        phi_min=0.20, phi_max=0.60, phi_ref=0.40,
        psi_min=0.15, psi_max=0.50, psi_ref=0.30,
        notes='Forward-curved sirocco fan. Lower efficiency. Good for HVAC at large flow.',
        warnings_at_limits=['Forward-curved blades: risk of motor overload at high flow'],
    ),
}


@dataclass
class PreliminaryDesign:
    """Result of preliminary design from database."""
    machine_type: str
    nq: float
    # Recommended geometry
    beta2_recommended_deg: float
    blade_count_recommended: int
    b2_d2_recommended: float
    phi_ref: float
    psi_ref: float
    eta_expected: float
    splitter_recommended: bool
    # Specific speed assessment
    nq_assessment: str          # 'below_range' | 'optimal' | 'above_range' | 'outside_range'
    nq_distance_from_opt: float # |Nq - Nq_opt| / (Nq_max - Nq_min)
    # Warnings
    warnings: list[str] = field(default_factory=list)
    notes: str = ''


def get_design_recommendation(
    machine_type: str,
    nq: float,
    blade_count_override: Optional[int] = None,
) -> PreliminaryDesign:
    """Get preliminary design recommendations for a machine type and Nq.

    Args:
        machine_type: Machine type key (see DESIGN_DATABASE)
        nq: Specific speed (Nq = N·√Q / H^0.75)
        blade_count_override: If given, override blade count recommendation

    Returns:
        PreliminaryDesign with recommended parameters
    """
    db = DESIGN_DATABASE.get(machine_type)
    if db is None:
        # Fall back to centrifugal_pump
        db = DESIGN_DATABASE['centrifugal_pump']

    warnings = list(db.warnings_at_limits) if nq <= db.nq_min or nq >= db.nq_max else []

    # Nq assessment
    if nq < db.nq_min:
        nq_assessment = 'below_range'
        warnings.append(f"Nq={nq:.0f} below typical range ({db.nq_min}-{db.nq_max}) for {machine_type}")
    elif nq > db.nq_max:
        nq_assessment = 'above_range'
        warnings.append(f"Nq={nq:.0f} above typical range ({db.nq_min}-{db.nq_max}) for {machine_type}")
    else:
        dist = abs(nq - db.nq_optimal) / max(1, db.nq_max - db.nq_min)
        nq_assessment = 'optimal' if dist < 0.15 else 'within_range'

    nq_dist = abs(nq - db.nq_optimal) / max(1, db.nq_max - db.nq_min)

    # Beta2 interpolation
    nq_frac = min(1.0, max(0.0, (nq - db.nq_min) / max(1.0, db.nq_max - db.nq_min)))
    beta2_rec = db.beta2_ref_deg + (nq_frac - 0.5) * (db.beta2_max_deg - db.beta2_min_deg) * 0.4
    beta2_rec = max(db.beta2_min_deg, min(db.beta2_max_deg, beta2_rec))

    # Blade count
    if blade_count_override is not None:
        z_rec = blade_count_override
    elif db.blade_count_formula == 'gulich':
        # Gülich Eq. 3.14: Z = 6.5 * (D2 + D1)/(D2 - D1) * sin((β1 + β2)/2)
        # Approximate with Nq-based estimate
        z_rec = max(db.blade_count_min, min(db.blade_count_max, round(7 - nq / 30)))
    elif db.blade_count_formula == 'pfleiderer':
        # Pfleiderer: Z = 6.5 * sin(β2_ref) * (r2 + r1)/(r2 - r1)
        z_rec = max(db.blade_count_min, min(db.blade_count_max, round(8 + nq / 20)))
    else:
        z_rec = round((db.blade_count_min + db.blade_count_max) / 2)

    # b2/D2
    b2_d2_rec = db.b2_d2_ratio_ref + db.b2_d2_nq_slope * (nq - db.nq_optimal)
    b2_d2_rec = max(0.02, min(0.60, b2_d2_rec))

    # Efficiency
    eta_exp = db.eta_ref + db.eta_nq_slope * (nq - db.nq_optimal)
    eta_exp = max(0.40, min(0.96, eta_exp))

    # Splitter recommendation
    splitter_rec = (machine_type == 'centrifugal_compressor' and nq > 35) or \
                   (machine_type == 'centrifugal_pump' and nq > 55)
    if splitter_rec:
        warnings.append(f"Splitter blades recommended for {machine_type} at Nq={nq:.0f}")

    return PreliminaryDesign(
        machine_type=machine_type,
        nq=nq,
        beta2_recommended_deg=round(beta2_rec, 1),
        blade_count_recommended=z_rec,
        b2_d2_recommended=round(b2_d2_rec, 4),
        phi_ref=db.phi_ref,
        psi_ref=db.psi_ref,
        eta_expected=round(eta_exp, 3),
        splitter_recommended=splitter_rec,
        nq_assessment=nq_assessment,
        nq_distance_from_opt=round(nq_dist, 3),
        warnings=warnings,
        notes=db.notes,
    )


def list_machine_types() -> list[dict]:
    """Return all machine types with their Nq ranges."""
    return [
        {
            'id': k,
            'nq_min': v.nq_min,
            'nq_max': v.nq_max,
            'nq_optimal': v.nq_optimal,
            'eta_ref': v.eta_ref,
            'notes': v.notes,
        }
        for k, v in DESIGN_DATABASE.items()
    ]
