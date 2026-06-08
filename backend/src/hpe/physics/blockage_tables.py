"""Blockage tables for turbomachinery blade passage analysis.

Blockage factor B(m) represents the fraction of the geometric passage
area that is available for through-flow after accounting for blade
thickness and boundary layers.  B = 1 means no blockage; B < 1 means
the effective area is reduced.

Classical formula:
    B(m) = 1 - z * t / (2 * pi * r * sin(beta))

where z = blade count, t = blade thickness, r = local radius,
beta = local blade angle.

Supports three construction modes:
    1. from_default — auto-compute from blade geometry
    2. from_two_control_points — linear LE-to-TE ramp (ADT style)
    3. from_table — user-specified 2-D table (m, s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BlockageTable:
    """Blockage distribution along the meridional (and optionally spanwise) direction.

    Attributes
    ----------
    m_points : NDArray
        Normalised meridional coordinate [0, 1] (LE=0, TE=1).
    s_points : NDArray
        Normalised spanwise coordinate [0, 1] (hub=0, shroud=1).
    values : NDArray
        2-D array of blockage factors, shape (len(m_points), len(s_points)).
        Each value is in (0, 1].
    """

    m_points: NDArray[np.float64]
    s_points: NDArray[np.float64]
    values: NDArray[np.float64]

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_default(
        cls,
        blade_count: int,
        thickness_dist: Sequence[float],
        blade_angles: Sequence[float],
        radii: Optional[Sequence[float]] = None,
        n_pts: int = 21,
    ) -> BlockageTable:
        """Auto-compute B(m) = 1 - z*t / (2*pi*r*sin(beta)).

        Parameters
        ----------
        blade_count : int
            Number of blades z.
        thickness_dist : sequence of float
            Blade thickness [m] at each meridional station.
        blade_angles : sequence of float
            Blade angle [deg] at each meridional station.
        radii : sequence of float, optional
            Local radius [m] at each station.  If *None*, a linear
            ramp from 0.05 to 0.15 m is assumed (typical small pump).
        n_pts : int
            Number of output meridional stations (resampled).

        Returns
        -------
        BlockageTable
        """
        t_arr = np.asarray(thickness_dist, dtype=np.float64)
        beta_arr = np.asarray(blade_angles, dtype=np.float64)
        n_input = len(t_arr)

        if radii is not None:
            r_arr = np.asarray(radii, dtype=np.float64)
        else:
            r_arr = np.linspace(0.05, 0.15, n_input)

        # Compute blockage at input stations
        beta_rad = np.radians(beta_arr)
        sin_beta = np.abs(np.sin(beta_rad))
        sin_beta = np.maximum(sin_beta, 1e-6)  # avoid division by zero

        z = blade_count
        b_raw = 1.0 - z * t_arr / (2.0 * math.pi * r_arr * sin_beta)
        b_raw = np.clip(b_raw, 0.01, 1.0)

        # Resample to uniform m grid
        m_in = np.linspace(0.0, 1.0, n_input)
        m_out = np.linspace(0.0, 1.0, n_pts)
        b_out = np.interp(m_out, m_in, b_raw)

        # 1-D table (single spanwise station at midspan)
        s_pts = np.array([0.5])
        vals = b_out.reshape(-1, 1)

        return cls(m_points=m_out, s_points=s_pts, values=vals)

    @classmethod
    def from_two_control_points(
        cls,
        b_inlet: float,
        b_outlet: float,
        n_pts: int = 21,
    ) -> BlockageTable:
        """Linear interpolation between inlet and outlet blockage (ADT style).

        Parameters
        ----------
        b_inlet : float
            Blockage factor at LE (e.g. 0.92).
        b_outlet : float
            Blockage factor at TE (e.g. 0.88).
        n_pts : int
            Number of meridional stations.

        Returns
        -------
        BlockageTable
        """
        m_pts = np.linspace(0.0, 1.0, n_pts)
        b_vals = np.linspace(b_inlet, b_outlet, n_pts)
        s_pts = np.array([0.5])
        vals = b_vals.reshape(-1, 1)
        return cls(m_points=m_pts, s_points=s_pts, values=vals)

    @classmethod
    def from_table(
        cls,
        m_points: Sequence[float],
        s_points: Sequence[float],
        values: Sequence[Sequence[float]],
    ) -> BlockageTable:
        """Construct from a user-specified 2-D table.

        Parameters
        ----------
        m_points : sequence of float
            Meridional coordinates [0, 1].
        s_points : sequence of float
            Spanwise coordinates [0, 1].
        values : 2-D sequence
            Blockage values, shape (len(m_points), len(s_points)).

        Returns
        -------
        BlockageTable
        """
        m_arr = np.asarray(m_points, dtype=np.float64)
        s_arr = np.asarray(s_points, dtype=np.float64)
        v_arr = np.asarray(values, dtype=np.float64)

        if v_arr.shape != (len(m_arr), len(s_arr)):
            raise ValueError(
                f"values shape {v_arr.shape} does not match "
                f"({len(m_arr)}, {len(s_arr)})"
            )

        return cls(m_points=m_arr, s_points=s_arr, values=v_arr)

    # -----------------------------------------------------------------
    # Interpolation
    # -----------------------------------------------------------------

    def interpolate(self, m_frac: float, s_frac: float = 0.5) -> float:
        """Bilinear interpolation of blockage at (m_frac, s_frac).

        Parameters
        ----------
        m_frac : float
            Normalised meridional position [0, 1].
        s_frac : float
            Normalised spanwise position [0, 1] (default midspan).

        Returns
        -------
        float
            Blockage factor.
        """
        m_frac = float(np.clip(m_frac, 0.0, 1.0))
        s_frac = float(np.clip(s_frac, 0.0, 1.0))

        # 1-D meridional if single spanwise station
        if len(self.s_points) == 1:
            return float(np.interp(m_frac, self.m_points, self.values[:, 0]))

        # 2-D bilinear: interpolate along m at each s, then along s
        row_vals = np.array([
            np.interp(m_frac, self.m_points, self.values[:, j])
            for j in range(len(self.s_points))
        ])
        return float(np.interp(s_frac, self.s_points, row_vals))

    def get_effective_area(self, m_frac: float, passage_area: float) -> float:
        """Return the effective through-flow area at a given meridional station.

        Parameters
        ----------
        m_frac : float
            Normalised meridional position [0, 1].
        passage_area : float
            Geometric passage area [m^2].

        Returns
        -------
        float
            Effective area = passage_area * blockage.
        """
        return passage_area * self.interpolate(m_frac)

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """Serialise for JSON transport."""
        return {
            "m_points": self.m_points.tolist(),
            "s_points": self.s_points.tolist(),
            "values": self.values.tolist(),
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PUMP_THIN = BlockageTable.from_two_control_points(0.92, 0.88)
PUMP_THICK = BlockageTable.from_two_control_points(0.85, 0.78)
COMPRESSOR = BlockageTable.from_two_control_points(0.90, 0.82)
TURBINE = BlockageTable.from_two_control_points(0.88, 0.85)

PRESETS: Dict[str, BlockageTable] = {
    "pump_thin": PUMP_THIN,
    "pump_thick": PUMP_THICK,
    "compressor": COMPRESSOR,
    "turbine": TURBINE,
}
