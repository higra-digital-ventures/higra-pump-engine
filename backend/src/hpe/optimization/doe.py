"""Design of Experiments — Optimal Latin Hypercube Sampling.

Generates space-filling design points for building surrogate models
before running expensive optimization.

References:
    McKay et al. (1979) — Latin Hypercube Sampling.
    Morris & Mitchell (1995) — Exploratory designs for computational experiments.
"""
from __future__ import annotations

import random
import math
from dataclasses import dataclass


@dataclass
class DoEConfig:
    """Latin Hypercube configuration."""

    n_points: int = 45          # Number of design points (like ADT: 45 OLH)
    n_variables: int = 4        # Number of design variables
    bounds: list[tuple[float, float]] = None  # [(lo, hi)] per variable
    seed: int = 42
    optimize_iterations: int = 100  # Iterations to improve space-filling


@dataclass
class DoEResult:
    """Result of Latin Hypercube sampling."""

    points: list[list[float]]   # [n_points][n_variables] design matrix
    min_distance: float          # Minimum inter-point distance (maximize this)
    coverage_metric: float       # Space-filling quality 0-1


def generate_lhs(config: DoEConfig) -> DoEResult:
    """Generate Optimal Latin Hypercube design.

    Uses random LHS then optimizes space-filling by swapping elements
    within columns (Morris-Mitchell criterion).
    """
    rng = random.Random(config.seed)
    n = config.n_points
    k = config.n_variables
    bounds = config.bounds or [(0.0, 1.0)] * k

    # Initial random LHS
    matrix = _init_lhs(n, k, rng)

    # Optimize using column-wise swaps (maximize min distance)
    best = matrix
    best_dist = _min_distance(best)

    for _ in range(config.optimize_iterations):
        # Randomly swap two elements in a random column
        col = rng.randint(0, k - 1)
        i, j = rng.sample(range(n), 2)
        candidate = [row[:] for row in matrix]
        candidate[i][col], candidate[j][col] = candidate[j][col], candidate[i][col]
        d = _min_distance(candidate)
        if d > best_dist:
            best_dist = d
            best = candidate
            matrix = candidate

    # Scale to bounds
    scaled = []
    for row in best:
        scaled_row = []
        for j, val in enumerate(row):
            lo, hi = bounds[j]
            scaled_row.append(lo + val * (hi - lo))
        scaled.append(scaled_row)

    coverage = min(1.0, best_dist * math.sqrt(k) / (n ** (1 / k)))

    return DoEResult(
        points=scaled,
        min_distance=round(best_dist, 6),
        coverage_metric=round(coverage, 4),
    )


def _init_lhs(n: int, k: int, rng: random.Random) -> list[list[float]]:
    """Initialize Latin Hypercube matrix."""
    # Each column: permutation of [0, n-1] scaled to [0, 1]
    matrix = [[0.0] * k for _ in range(n)]
    for col in range(k):
        perm = list(range(n))
        rng.shuffle(perm)
        for row in range(n):
            matrix[row][col] = (perm[row] + rng.random()) / n
    return matrix


def _min_distance(matrix: list[list[float]]) -> float:
    """Compute minimum Euclidean distance between any two points."""
    n = len(matrix)
    if n < 2:
        return 1.0
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                sum((matrix[i][k] - matrix[j][k]) ** 2 for k in range(len(matrix[i])))
            )
            if d < min_d:
                min_d = d
                if min_d < 1e-10:
                    return min_d
    return min_d
