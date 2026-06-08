"""Design of Experiments (DoE) — amostragem do espaço de projeto — Fase 15.

Gera planos de experimento para varredura paramétrica multi-variável das
variáveis de projeto da bomba centrífuga.

Métodos implementados:
  - Latin Hypercube Sampling (LHS) — distribuição uniforme com cobertura garantida
  - Full factorial (grid) — para espaços pequenos (≤ 3 variáveis)
  - Random uniform — fallback sem dependência externa

Usage
-----
    from hpe.cfd.doe import generate_lhs, DesignSpace, DesignPoint

    space = DesignSpace.from_sizing(sizing_result, variation=0.15)
    points = generate_lhs(space, n_samples=20)
    for pt in points:
        print(pt.beta2, pt.d2, pt.b2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Espaço de projeto
# ---------------------------------------------------------------------------

@dataclass
class VariableBounds:
    """Limites de uma variável de projeto."""
    name: str
    lo: float
    hi: float
    nominal: float
    unit: str = ""

    def scale_01(self, value: float) -> float:
        """Normalizar para [0, 1]."""
        rng = self.hi - self.lo
        return (value - self.lo) / rng if rng > 0 else 0.5

    def unscale(self, u: float) -> float:
        """Converter de [0, 1] para espaço original."""
        return self.lo + u * (self.hi - self.lo)


@dataclass
class DesignSpace:
    """Espaço de busca multidimensional para otimização/DoE.

    Attributes
    ----------
    variables : list[VariableBounds]
        Variáveis de projeto com seus limites.
    """
    variables: list[VariableBounds] = field(default_factory=list)

    @property
    def ndim(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @classmethod
    def from_sizing(
        cls,
        sizing_result,
        variation: float = 0.15,
        include: Optional[list[str]] = None,
    ) -> "DesignSpace":
        """Construir DesignSpace a partir de um SizingResult.

        Parameters
        ----------
        sizing_result : SizingResult
            Ponto nominal de referência.
        variation : float
            Variação relativa (±) em torno do nominal.  0.15 = ±15%.
        include : list[str] | None
            Variáveis a incluir.  None = todas as padrão.
            Opções: "beta1", "beta2", "d2", "d1", "b2", "blade_count"
        """
        if include is None:
            include = ["beta1", "beta2", "d2", "b2"]

        # Extrair valores nominais do sizing_result
        def get(attr: str, default: float) -> float:
            return float(getattr(sizing_result, attr, default))

        nominals = {
            "beta1":       (get("beta1", 25.0),  10.0, 45.0,  "deg"),
            "beta2":       (get("beta2", 22.0),  12.0, 40.0,  "deg"),
            "d2":          (get("d2",    0.30),   0.05, 1.0,   "m"),
            "d1":          (get("d1",    0.15),   0.02, 0.5,   "m"),
            "b2":          (get("b2",    0.02),   0.005, 0.2,  "m"),
            "blade_count": (get("blade_count", 6), 4,   9,     ""),
        }

        variables = []
        for name in include:
            if name not in nominals:
                continue
            nom, phys_lo, phys_hi, unit = nominals[name]
            lo = max(phys_lo, nom * (1 - variation))
            hi = min(phys_hi, nom * (1 + variation))
            variables.append(VariableBounds(
                name=name, lo=lo, hi=hi, nominal=nom, unit=unit
            ))

        return cls(variables=variables)

    def point_from_array(self, x: "np.ndarray") -> "DesignPoint":
        """Converter array [0,1]^n em DesignPoint."""
        values = {
            v.name: v.unscale(float(x[i]))
            for i, v in enumerate(self.variables)
        }
        return DesignPoint(values=values)

    def to_dict(self) -> dict:
        return {
            "ndim": self.ndim,
            "variables": [
                {
                    "name": v.name,
                    "lo": v.lo,
                    "hi": v.hi,
                    "nominal": v.nominal,
                    "unit": v.unit,
                }
                for v in self.variables
            ],
        }


@dataclass
class DesignPoint:
    """Um ponto no espaço de projeto.

    Attributes
    ----------
    values : dict[str, float]
        Valor de cada variável de projeto.
    """
    values: dict[str, float]

    def __getattr__(self, name: str) -> float:
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.values.items()}


# ---------------------------------------------------------------------------
# Geradores de planos
# ---------------------------------------------------------------------------

def generate_lhs(
    space: DesignSpace,
    n_samples: int,
    seed: Optional[int] = None,
) -> list[DesignPoint]:
    """Latin Hypercube Sampling — distribuição uniforme estratificada.

    Garante que cada estrato [k/n, (k+1)/n] seja amostrado exatamente
    uma vez por dimensão.

    Parameters
    ----------
    space : DesignSpace
        Espaço de busca.
    n_samples : int
        Número de pontos (≥ 2).
    seed : int | None
        Semente para reprodutibilidade.

    Returns
    -------
    list[DesignPoint]
        Lista de pontos amostrados.
    """
    rng = np.random.default_rng(seed)
    n, d = n_samples, space.ndim

    # LHS: para cada dimensão, permutação aleatória de estratos
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n  # uniforme dentro de cada estrato
        samples[:, j] = u

    return [space.point_from_array(samples[i]) for i in range(n)]


def generate_full_factorial(
    space: DesignSpace,
    levels: int = 3,
) -> list[DesignPoint]:
    """Full factorial: grade regular com ``levels`` pontos por dimensão.

    Atenção: número de pontos = levels^ndim.  Usar apenas para ndim ≤ 4.

    Parameters
    ----------
    space : DesignSpace
        Espaço de busca.
    levels : int
        Número de níveis por variável (2=extremos, 3=extremos+centro, etc.).
    """
    total = levels ** space.ndim
    if total > 10000:
        raise ValueError(
            f"Full factorial com {levels}^{space.ndim} = {total} pontos é excessivo. "
            "Use generate_lhs() para espaços de alta dimensão."
        )

    grid_1d = np.linspace(0, 1, levels)
    grids = np.meshgrid(*[grid_1d] * space.ndim, indexing="ij")
    flat = [g.ravel() for g in grids]
    x_all = np.column_stack(flat) if len(flat) > 1 else flat[0].reshape(-1, 1)

    return [space.point_from_array(x_all[i]) for i in range(len(x_all))]


def generate_random(
    space: DesignSpace,
    n_samples: int,
    seed: Optional[int] = None,
) -> list[DesignPoint]:
    """Amostragem aleatória uniforme (fallback sem NumPy avançado)."""
    rng = np.random.default_rng(seed)
    samples = rng.random((n_samples, space.ndim))
    return [space.point_from_array(samples[i]) for i in range(n_samples)]


def generate_sobol(
    space: DesignSpace,
    n_samples: int,
    seed: Optional[int] = None,
) -> list[DesignPoint]:
    """Sequência de Sobol (quasi-random) — melhor cobertura que LHS para ndim > 6.

    Requer scipy >= 1.7.  Cai em LHS se não disponível.
    """
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=space.ndim, scramble=True, seed=seed)
        samples = sampler.random(n_samples)
        return [space.point_from_array(samples[i]) for i in range(n_samples)]
    except ImportError:
        log.warning("scipy.stats.qmc.Sobol não disponível; usando LHS como fallback")
        return generate_lhs(space, n_samples, seed)
