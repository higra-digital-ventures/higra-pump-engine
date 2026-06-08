"""ANSYS Fluent results parser — extracts performance from report/monitor exports.

Parses Fluent report files, surface-integral exports, and XY plot data
to compute pump performance metrics (head, efficiency, power).

References:
    ANSYS Fluent User's Guide — report file format.
    ANSYS Fluent User's Guide — XY plot export format.
"""

from __future__ import annotations

import math
import re
from typing import Any


G = 9.80665  # m/s^2


class FluentResultsParser:
    """Parses ANSYS Fluent report and plot exports and computes performance."""

    def parse_report(self, filepath_or_content: str) -> dict[str, Any]:
        """Parse a Fluent report/monitor output file.

        Fluent report files typically contain iteration-indexed data
        with columns for residuals, surface integrals, and custom
        report definitions.  This method accepts either a file path
        or raw text content.

        Args:
            filepath_or_content: Path to the report file, or its raw
                text content (auto-detected by checking for newlines).

        Returns:
            Dictionary with keys:
                - iteration: list[int]
                - residuals: dict[str, list[float]]
                - surface_integrals: dict[str, float] — last values
        """
        if "\n" in filepath_or_content:
            content = filepath_or_content
        else:
            with open(filepath_or_content, "r", encoding="utf-8") as fh:
                content = fh.read()

        iterations: list[int] = []
        residuals: dict[str, list[float]] = {}
        surface_integrals: dict[str, float] = {}

        lines = content.strip().splitlines()

        # Try to detect CSV-style monitor output
        if lines and ("," in lines[0] or "\t" in lines[0]):
            return self._parse_csv_report(lines)

        # Parse free-form Fluent report output
        current_section = ""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("("):
                continue

            # Detect iteration lines: "  iter  continuity  x-velocity ..."
            if stripped.lower().startswith("iter"):
                current_section = "residuals"
                header_parts = stripped.split()
                for part in header_parts[1:]:
                    residuals.setdefault(part, [])
                continue

            # Parse residual data lines
            if current_section == "residuals":
                parts = stripped.split()
                if parts and parts[0].isdigit():
                    iterations.append(int(parts[0]))
                    header_keys = list(residuals.keys())
                    for i, key in enumerate(header_keys):
                        if i + 1 < len(parts):
                            try:
                                residuals[key].append(float(parts[i + 1]))
                            except ValueError:
                                residuals[key].append(0.0)
                    continue

            # Detect surface integral results
            # e.g., "Area-Weighted Average of Pressure on inlet = 12345.6 (Pa)"
            match = re.match(
                r".*(?:Average|Integral|Total|Rate)\s+.*?on\s+(\S+)\s*=\s*([-\d.eE+]+)",
                stripped,
                re.IGNORECASE,
            )
            if match:
                surface_name = match.group(1).strip()
                value = float(match.group(2))
                surface_integrals[surface_name] = value
                continue

            # Detect "key = value" patterns
            kv_match = re.match(r"(\S[\w\s-]+\S)\s*[:=]\s*([-\d.eE+]+)", stripped)
            if kv_match:
                key = kv_match.group(1).strip().lower().replace(" ", "_")
                surface_integrals[key] = float(kv_match.group(2))

        return {
            "iteration": iterations,
            "residuals": residuals,
            "surface_integrals": surface_integrals,
        }

    def parse_xy_plot(self, filepath_or_content: str) -> dict[str, Any]:
        """Parse a Fluent XY plot export (e.g., blade loading).

        XY plot exports are tab or space-delimited with two columns
        (typically position vs. pressure or velocity).

        Args:
            filepath_or_content: Path to the XY file, or its raw
                text content.

        Returns:
            Dictionary with keys:
                - x: list[float] — independent variable (e.g., chord position)
                - y: list[float] — dependent variable (e.g., pressure)
                - title: str — plot title if found in header
        """
        if "\n" in filepath_or_content:
            content = filepath_or_content
        else:
            with open(filepath_or_content, "r", encoding="utf-8") as fh:
                content = fh.read()

        x_vals: list[float] = []
        y_vals: list[float] = []
        title = ""

        for line in content.strip().splitlines():
            stripped = line.strip()

            # Skip comment/header lines
            if stripped.startswith(("#", ";", "(")):
                if "title" in stripped.lower() or not title:
                    title = stripped.lstrip("#;( ").rstrip(") ")
                continue

            # Parse data lines (tab or space delimited)
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                except ValueError:
                    continue

        return {
            "x": x_vals,
            "y": y_vals,
            "title": title,
        }

    def compute_performance(
        self,
        parsed: dict[str, Any],
        rpm: float,
        fluid_density: float = 998.0,
    ) -> dict[str, Any]:
        """Compute pump performance from parsed Fluent report data.

        Args:
            parsed: Output of parse_report.
            rpm: Rotational speed [rev/min], needed for shaft power.
            fluid_density: Fluid density [kg/m3].

        Returns:
            Dictionary with keys:
                - head_m: Pump head [m]
                - efficiency: Hydraulic efficiency [-]
                - power_w: Shaft power [W]
        """
        si = parsed.get("surface_integrals", {})
        omega = rpm * math.pi / 30.0

        # Extract pressures (try common key patterns)
        p_in = 0.0
        p_out = 0.0
        m_dot = 0.0
        torque_val = 0.0

        for key, val in si.items():
            key_lower = key.lower()
            if "inlet" in key_lower and "pressure" in key_lower:
                p_in = val
            elif "outlet" in key_lower and "pressure" in key_lower:
                p_out = val
            elif "inlet" in key_lower:
                # Could be pressure on inlet surface
                if p_in == 0.0:
                    p_in = val
            elif "outlet" in key_lower:
                if p_out == 0.0:
                    p_out = val
            elif "mass" in key_lower or "flow" in key_lower:
                m_dot = abs(val)
            elif "torque" in key_lower or "moment" in key_lower:
                torque_val = abs(val)

        delta_p = p_out - p_in
        head_m = delta_p / (fluid_density * G) if fluid_density > 0 else 0.0

        # Hydraulic power
        p_hydraulic = m_dot * G * head_m

        # Shaft power
        power_w = torque_val * omega if omega > 0 else 0.0

        # Efficiency
        efficiency = 0.0
        if power_w > 0:
            efficiency = min(1.0, abs(p_hydraulic) / power_w)

        return {
            "head_m": round(head_m, 4),
            "efficiency": round(efficiency, 4),
            "power_w": round(power_w, 2),
            "pressure_rise_pa": round(delta_p, 2),
            "mass_flow_kgs": round(m_dot, 6),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_csv_report(self, lines: list[str]) -> dict[str, Any]:
        """Parse CSV-formatted report data."""
        iterations: list[int] = []
        residuals: dict[str, list[float]] = {}
        surface_integrals: dict[str, float] = {}

        # Detect delimiter
        delimiter = "," if "," in lines[0] else "\t"
        header = [col.strip().strip('"') for col in lines[0].split(delimiter)]

        for col_name in header[1:]:
            residuals[col_name] = []

        for line in lines[1:]:
            parts = [p.strip().strip('"') for p in line.split(delimiter)]
            if not parts or not parts[0]:
                continue
            try:
                iterations.append(int(float(parts[0])))
                for i, col_name in enumerate(header[1:], start=1):
                    if i < len(parts):
                        residuals[col_name].append(float(parts[i]))
            except ValueError:
                continue

        # Store last values as surface integrals for convenience
        for key, vals in residuals.items():
            if vals:
                surface_integrals[key] = vals[-1]

        return {
            "iteration": iterations,
            "residuals": residuals,
            "surface_integrals": surface_integrals,
        }
