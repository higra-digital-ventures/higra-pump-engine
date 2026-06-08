"""Comprehensive noise prediction for hydraulic turbomachinery.

Implements broadband, tonal (BPF), and cavitation noise models for
centrifugal pumps.  The combined sound power level is computed with
optional A-weighting.

Models:
    1. Broadband — simplified Brooks-Pope-Marcolini (BPM) turbulent
       boundary-layer self-noise.
    2. Tonal — blade-passing frequency and harmonics from loading
       fluctuation.
    3. Cavitation — onset and growth based on Thoma cavitation index.

References:
    - Brooks, Pope & Marcolini (1989). Airfoil self-noise and prediction.
    - Langthjem & Olhoff (2004). Pump noise analysis.
    - Guelich & Bolleter (1992). Pressure pulsations in centrifugal pumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# A-weighting curve (IEC 61672)
# ---------------------------------------------------------------------------

def _a_weight(f: float) -> float:
    """Return A-weighting correction in dB for frequency *f* [Hz].

    Args:
        f: Frequency in Hz (must be > 0).

    Returns:
        A-weighting correction in dB.
    """
    if f <= 0:
        return -100.0
    f2 = f * f
    num = 12194.0 ** 2 * f2 * f2
    den = (
        (f2 + 20.6 ** 2)
        * math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    if den == 0:
        return -100.0
    ra = num / den
    return 20.0 * math.log10(max(ra, 1e-20)) + 2.0


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class NoiseResult:
    """Full noise prediction output."""

    lw_total_dB: float                   # Overall sound power level [dB re 1 pW]
    lw_A_weighted_dB: float              # A-weighted overall [dB(A)]
    spectrum: List[Dict[str, float]]     # [{frequency_hz, lw_dB, lw_A_dB}, ...]
    bpf_hz: float                        # Blade-passing frequency [Hz]
    dominant_source: str                 # "broadband" | "tonal" | "cavitation"
    lw_broadband_dB: float               # Broadband component [dB]
    lw_tonal_dB: float                   # Tonal (BPF + harmonics) [dB]
    lw_cavitation_dB: float              # Cavitation noise [dB]
    harmonics: List[Dict[str, float]]    # BPF harmonics [{order, freq_hz, lw_dB}]
    cavitation_onset: bool               # True if sigma < sigma_i
    warnings: List[str] = field(default_factory=list)

    def is_above_limit(self, limit_dB: float) -> bool:
        """Check if total noise exceeds a given limit.

        Args:
            limit_dB: Sound power level limit [dB].

        Returns:
            True if lw_total_dB > limit_dB.
        """
        return self.lw_total_dB > limit_dB


# ---------------------------------------------------------------------------
# Noise predictor
# ---------------------------------------------------------------------------

class NoisePredictor:
    """Multi-source noise prediction for centrifugal pumps.

    Combines broadband (BPM-simplified), tonal (BPF harmonics),
    and cavitation noise into an overall spectrum.
    """

    N_HARMONICS: int = 3
    N_SPECTRUM_POINTS: int = 64

    def __init__(
        self,
        rho: float = 998.2,
        c_sound: float = 1480.0,
        p_vapor: float = 2340.0,
        p_atm: float = 101325.0,
        mu: float = 1.003e-3,
    ) -> None:
        """Initialise with fluid properties.

        Args:
            rho: Fluid density [kg/m^3] (default: water at 20 C).
            c_sound: Speed of sound in fluid [m/s].
            p_vapor: Vapor pressure [Pa].
            p_atm: Atmospheric pressure [Pa].
            mu: Dynamic viscosity [Pa s].
        """
        self.rho = rho
        self.c_sound = c_sound
        self.p_vapor = p_vapor
        self.p_atm = p_atm
        self.mu = mu

    # ----- broadband noise (simplified BPM) -----

    def _broadband(
        self,
        u_tip: float,
        d2: float,
        b2: float,
        blade_count: int,
        rpm: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Broadband turbulent boundary-layer noise.

        Simplified from the BPM model:
            Lw_bb = 10*log10(rho * c^5 * A_blade / r^2) + f(Re, Ma)

        Args:
            u_tip: Tip speed [m/s].
            d2: Impeller outer diameter [m].
            b2: Outlet width [m].
            blade_count: Number of blades.
            rpm: Rotational speed [rev/min].

        Returns:
            Tuple of (Lw_bb total [dB], frequency array [Hz], spectrum [dB]).
        """
        # Blade chord estimate ≈ (D2 - D1)/2 but we approximate from b2 and d2
        chord_est = max(b2 * 1.5, d2 * 0.15)
        a_blade = chord_est * b2 * blade_count  # total blade area [m^2]

        # Reference distance (1 m for sound power level)
        r_ref = 1.0

        # Reynolds and Mach
        re_c = self.rho * u_tip * chord_est / self.mu
        ma = u_tip / self.c_sound

        # BPM simplified: Lw = 10*log10(rho * c^5 * A / r^2) + 10*log10(Ma^5) + Re correction
        base = 10.0 * math.log10(
            max(self.rho * self.c_sound ** 5 * a_blade / (r_ref ** 2), 1e-30)
        )
        ma_term = 50.0 * math.log10(max(ma, 1e-6))
        re_term = -5.0 * math.log10(max(re_c / 1e6, 0.1))

        lw_bb = base + ma_term + re_term

        # Spectrum shape: broadband with peak around 0.3 * u_tip / chord
        f_peak = 0.3 * u_tip / chord_est
        freqs = np.logspace(1.0, 4.5, self.N_SPECTRUM_POINTS)
        # Gaussian shape in log-frequency space
        log_ratio = np.log10(freqs / f_peak)
        shape = np.exp(-0.5 * (log_ratio / 0.4) ** 2)
        spectrum_db = lw_bb + 10.0 * np.log10(np.maximum(shape, 1e-20))

        return float(lw_bb), freqs, spectrum_db

    # ----- tonal noise (BPF) -----

    def _tonal(
        self,
        blade_count: int,
        rpm: float,
        u_tip: float,
        d2: float,
        b2: float,
        efficiency: float,
    ) -> Tuple[float, float, List[Dict[str, float]], np.ndarray, np.ndarray]:
        """Blade-passing frequency tonal noise.

        BPF = z * n / 60.  Amplitude scales with blade loading and tip
        speed.

        Args:
            blade_count: Number of blades.
            rpm: Rotational speed [rev/min].
            u_tip: Tip speed [m/s].
            d2: Impeller outer diameter [m].
            b2: Outlet width [m].
            efficiency: Estimated total efficiency.

        Returns:
            Tuple of (Lw_tonal [dB], bpf [Hz], harmonics list,
            frequency array, spectrum array).
        """
        bpf = blade_count * rpm / 60.0

        # Loading factor: higher loading → more tonal noise
        loading_factor = (1.0 - efficiency) * 2.0  # rough proxy

        # Base tonal level from tip speed
        lw_base = 20.0 * math.log10(max(u_tip, 1.0)) + 10.0 * math.log10(
            max(d2 * b2 * blade_count, 1e-6)
        ) + 40.0

        harmonics: List[Dict[str, float]] = []
        harmonic_powers: list[float] = []

        for k in range(1, self.N_HARMONICS + 1):
            f_k = k * bpf
            # Each harmonic drops ~6 dB per order
            lw_k = lw_base + 20.0 * math.log10(loading_factor + 0.1) - 6.0 * (k - 1)
            harmonics.append({
                "order": float(k),
                "freq_hz": f_k,
                "lw_dB": lw_k,
            })
            harmonic_powers.append(10.0 ** (lw_k / 10.0))

        lw_tonal = 10.0 * math.log10(max(sum(harmonic_powers), 1e-30))

        # Tonal spectrum: narrow peaks at BPF harmonics
        freqs = np.logspace(1.0, 4.5, self.N_SPECTRUM_POINTS)
        spectrum = np.full_like(freqs, -200.0)
        for h in harmonics:
            f_h = h["freq_hz"]
            lw_h = h["lw_dB"]
            # Narrow Gaussian peak (Q ~ 50)
            peak = lw_h * np.exp(-0.5 * ((freqs - f_h) / (f_h * 0.02)) ** 2)
            spectrum = np.where(peak > spectrum, peak, spectrum)

        return float(lw_tonal), bpf, harmonics, freqs, spectrum

    # ----- cavitation noise -----

    def _cavitation(
        self,
        sigma: float,
        blade_count: int,
        rpm: float,
        u_tip: float,
    ) -> Tuple[float, bool, np.ndarray, np.ndarray]:
        """Cavitation noise estimate.

        Onset when sigma < sigma_i (incipient cavitation number).
        Sound power level grows with (sigma_i - sigma)^2.

        Args:
            sigma: Thoma cavitation number for the operating point.
            blade_count: Number of blades.
            rpm: Rotational speed [rev/min].
            u_tip: Tip speed [m/s].

        Returns:
            Tuple of (Lw_cav [dB], onset bool, frequency array, spectrum).
        """
        # Incipient cavitation number (empirical, depends on design)
        sigma_i = 0.3 + 0.5 * (u_tip / 30.0) ** 0.5

        freqs = np.logspace(1.0, 4.5, self.N_SPECTRUM_POINTS)
        onset = sigma < sigma_i

        if not onset:
            return -200.0, False, freqs, np.full_like(freqs, -200.0)

        # Severity factor
        severity = min((sigma_i - sigma) / sigma_i, 1.0)

        # Characteristic cavitation frequency
        f_cav = 0.5 * rpm / 60.0 * blade_count

        # Sound power: grows with severity^2 and u_tip^6
        lw_cav = (
            10.0 * math.log10(max(severity ** 2, 1e-20))
            + 60.0 * math.log10(max(u_tip, 1.0))
            - 60.0
        )

        # Broadband spectrum peaked at f_cav
        log_ratio = np.log10(np.maximum(freqs, 1.0) / max(f_cav, 1.0))
        shape = np.exp(-0.5 * (log_ratio / 0.5) ** 2)
        spectrum = lw_cav + 10.0 * np.log10(np.maximum(shape, 1e-20))

        return float(lw_cav), True, freqs, spectrum

    # ----- main prediction -----

    def predict(
        self,
        sizing_result: Any,
        rpm: float,
        fluid: str = "water",
    ) -> NoiseResult:
        """Run full noise prediction for a centrifugal pump.

        Args:
            sizing_result: SizingResult dataclass from meanline sizing.
            rpm: Rotational speed [rev/min].
            fluid: Fluid name (currently only affects density/sound speed
                   through constructor defaults).

        Returns:
            NoiseResult with total, A-weighted levels and spectrum.
        """
        warnings: list[str] = []

        d2 = sizing_result.impeller_d2
        b2 = sizing_result.impeller_b2
        blade_count = sizing_result.blade_count
        efficiency = sizing_result.estimated_efficiency
        sigma = sizing_result.sigma

        u_tip = math.pi * d2 * rpm / 60.0
        ma = u_tip / self.c_sound
        if ma > 0.3:
            warnings.append(
                f"Tip Mach {ma:.3f} > 0.3: compressibility effects may be significant"
            )

        # Individual sources
        lw_bb, bb_freqs, bb_spec = self._broadband(u_tip, d2, b2, blade_count, rpm)
        lw_ton, bpf, harmonics, ton_freqs, ton_spec = self._tonal(
            blade_count, rpm, u_tip, d2, b2, efficiency
        )
        lw_cav, cav_onset, cav_freqs, cav_spec = self._cavitation(
            sigma, blade_count, rpm, u_tip
        )

        if cav_onset:
            warnings.append("Cavitation noise present — consider increasing NPSH margin")

        # Combined spectrum (same frequency axis)
        freqs = bb_freqs  # all use the same grid
        combined_power = (
            10.0 ** (bb_spec / 10.0)
            + 10.0 ** (ton_spec / 10.0)
            + 10.0 ** (cav_spec / 10.0)
        )
        combined_db = 10.0 * np.log10(np.maximum(combined_power, 1e-30))

        # A-weighting
        a_corrections = np.array([_a_weight(f) for f in freqs])
        combined_a_db = combined_db + a_corrections

        # Overall levels
        lw_total = 10.0 * math.log10(
            10.0 ** (lw_bb / 10.0)
            + 10.0 ** (lw_ton / 10.0)
            + (10.0 ** (lw_cav / 10.0) if lw_cav > -100 else 0.0)
        )

        # A-weighted total from spectrum
        lw_a_total = 10.0 * math.log10(
            max(np.sum(10.0 ** (combined_a_db / 10.0)), 1e-30)
        )

        # Dominant source
        source_levels = {"broadband": lw_bb, "tonal": lw_ton, "cavitation": lw_cav}
        dominant = max(source_levels, key=source_levels.get)  # type: ignore[arg-type]

        # Build spectrum list
        spectrum: List[Dict[str, float]] = []
        for i, f in enumerate(freqs):
            spectrum.append({
                "frequency_hz": float(f),
                "lw_dB": float(combined_db[i]),
                "lw_A_dB": float(combined_a_db[i]),
            })

        return NoiseResult(
            lw_total_dB=float(lw_total),
            lw_A_weighted_dB=float(lw_a_total),
            spectrum=spectrum,
            bpf_hz=float(bpf),
            dominant_source=dominant,
            lw_broadband_dB=float(lw_bb),
            lw_tonal_dB=float(lw_ton),
            lw_cavitation_dB=float(lw_cav),
            harmonics=harmonics,
            cavitation_onset=cav_onset,
            warnings=warnings,
        )
