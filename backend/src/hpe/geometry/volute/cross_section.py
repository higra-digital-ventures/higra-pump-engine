"""Volute cross-section profile generation.

Creates 2D cross-section profiles (wires) at each circumferential
station, which are then lofted to create the 3D volute.
"""

from __future__ import annotations

import math


def circular_section(
    area: float,
    center_r: float,
    center_z: float = 0.0,
    n_points: int = 20,
) -> list[tuple[float, float]]:
    """Generate a circular cross-section profile.

    Args:
        area: Cross-section area [m^2].
        center_r: Radial position of section center [m].
        center_z: Axial position of section center [m].
        n_points: Number of points on the circle.

    Returns:
        List of (r, z) points defining the section outline.
    """
    if area <= 0:
        return [(center_r, center_z)]

    radius = math.sqrt(area / math.pi)
    points: list[tuple[float, float]] = []

    for i in range(n_points + 1):
        angle = 2.0 * math.pi * i / n_points
        r = center_r + radius * math.cos(angle)
        z = center_z + radius * math.sin(angle)
        points.append((r, z))

    return points


def trapezoidal_section(
    area: float,
    height: float,
    center_r: float,
    center_z: float = 0.0,
    taper_ratio: float = 0.7,
) -> list[tuple[float, float]]:
    """Generate a trapezoidal cross-section.

    Args:
        area: Cross-section area [m^2].
        height: Section height (typically b2) [m].
        center_r: Radial position [m].
        center_z: Axial position [m].
        taper_ratio: Top width / bottom width ratio.

    Returns:
        List of (r, z) points.
    """
    if area <= 0 or height <= 0:
        return [(center_r, center_z)]

    # A = h * (w_bottom + w_top) / 2 = h * w_bottom * (1 + taper) / 2
    w_bottom = 2.0 * area / (height * (1.0 + taper_ratio))
    w_top = w_bottom * taper_ratio

    hz = height / 2.0

    return [
        (center_r, center_z - hz),
        (center_r + w_bottom, center_z - hz),
        (center_r + w_top, center_z + hz),
        (center_r, center_z + hz),
        (center_r, center_z - hz),  # Close
    ]


def rectangular_section(
    area: float,
    height: float,
    center_r: float,
    center_z: float = 0.0,
) -> list[tuple[float, float]]:
    """Generate a rectangular cross-section.

    Args:
        area: Cross-section area [m^2].
        height: Section height [m].
        center_r: Radial position [m].
        center_z: Axial position [m].

    Returns:
        List of (r, z) points.
    """
    if area <= 0 or height <= 0:
        return [(center_r, center_z)]

    width = area / height
    hz = height / 2.0

    return [
        (center_r, center_z - hz),
        (center_r + width, center_z - hz),
        (center_r + width, center_z + hz),
        (center_r, center_z + hz),
        (center_r, center_z - hz),  # Close
    ]
