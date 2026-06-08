"""Prism layers automáticos com y+ targeting — Fase 17.3.

Gera configuração de `addLayersControls` para snappyHexMesh que atinge
um y+ alvo automaticamente.  Calcula:
  - Espessura da primeira camada (via compute_first_cell_height)
  - Número de camadas para cobrir a boundary layer (δ99)
  - Expansion ratio

Esse módulo é equivalente à aba "Inflation" do Ansys Meshing.

Usage
-----
    from hpe.cfd.mesh.prism_layers import compute_prism_layer_config

    cfg = compute_prism_layer_config(
        u_ref=10.0, l_ref=0.3, nu=1e-6,
        target_yplus=1.0,  # y+ target para k-ω SST low-Re
    )
    print(cfg.first_layer_thickness, cfg.n_layers, cfg.expansion_ratio)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .yplus import compute_first_cell_height


@dataclass
class PrismLayerConfig:
    """Configuração de camadas prismáticas para snappy.

    Attributes
    ----------
    n_layers : int
        Número de camadas (tipicamente 10-20 para SST).
    first_layer_thickness : float
        Espessura da primeira camada [m].
    expansion_ratio : float
        Razão de expansão entre camadas sucessivas (1.1-1.3).
    total_thickness : float
        Espessura total = Σ first*ratio^i.
    final_layer_thickness : float
        Espessura da última camada (deve ser próxima ao tamanho do cell core).
    yplus_target : float
        y+ alvo usado no cálculo.
    yplus_achieved : float
        y+ efetivamente atingido (verification).
    delta99_estimate : float
        Estimativa da espessura da boundary layer [m].
    """
    n_layers: int
    first_layer_thickness: float
    expansion_ratio: float
    total_thickness: float
    final_layer_thickness: float
    yplus_target: float
    yplus_achieved: float
    delta99_estimate: float

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "first_layer_thickness": round(self.first_layer_thickness, 8),
            "expansion_ratio": round(self.expansion_ratio, 3),
            "total_thickness": round(self.total_thickness, 6),
            "final_layer_thickness": round(self.final_layer_thickness, 6),
            "yplus_target": self.yplus_target,
            "yplus_achieved": round(self.yplus_achieved, 2),
            "delta99_estimate": round(self.delta99_estimate, 6),
        }

    def to_snappy_dict_entry(self, cell_core_size: float = 0.005) -> str:
        """Gerar bloco ``addLayersControls`` para snappyHexMeshDict.

        Parameters
        ----------
        cell_core_size : float
            Tamanho característico dos elementos do core mesh (para
            calcular ``finalLayerThickness`` como fração).
        """
        final_rel = max(0.3, min(1.0, self.final_layer_thickness / cell_core_size))
        return f"""\
addLayersControls
{{
    relativeSizes       false;
    layers
    {{
        "(blade.*|hub|shroud|walls)"
        {{
            nSurfaceLayers  {self.n_layers};
        }}
    }}

    expansionRatio          {self.expansion_ratio:.3f};
    firstLayerThickness     {self.first_layer_thickness:.6e};
    minThickness            {self.first_layer_thickness * 0.5:.6e};
    finalLayerThickness     {final_rel:.3f};

    nGrow                   0;

    featureAngle            120;
    slipFeatureAngle        30;
    nRelaxIter              5;
    nSmoothSurfaceNormals   3;
    nSmoothNormals          3;
    nSmoothThickness        10;
    maxFaceThicknessRatio   0.5;
    maxThicknessToMedialRatio   0.3;
    minMedianAxisAngle      90;
    nBufferCellsNoExtrude   0;
    nLayerIter              50;
    nRelaxedIter            20;
}}
"""


def compute_prism_layer_config(
    u_ref: float,
    l_ref: float,
    nu: float = 1e-6,
    rho: float = 998.2,
    target_yplus: float = 1.0,
    expansion_ratio: float = 1.2,
    coverage_fraction: float = 1.0,
) -> PrismLayerConfig:
    """Calcular configuração de prism layers para atingir y+ alvo.

    Parameters
    ----------
    u_ref : float
        Velocidade de referência (típico: u2 = π D2 n / 60) [m/s].
    l_ref : float
        Comprimento característico (corda da pá) [m].
    nu : float
        Viscosidade cinemática [m²/s] (água @20°C = 1e-6).
    rho : float
        Densidade [kg/m³].
    target_yplus : float
        y+ alvo.  Use 1 para low-Re SST, 30 para wall functions.
    expansion_ratio : float
        Razão de crescimento entre camadas (1.1-1.3 recomendado).
    coverage_fraction : float
        Fração da boundary layer δ99 a cobrir (1.0 = 100%).

    Returns
    -------
    PrismLayerConfig
        Configuração completa incluindo n_layers e total_thickness.
    """
    # Espessura da primeira camada para o y+ alvo
    yplus_est = compute_first_cell_height(
        u_ref=u_ref, l_ref=l_ref, nu=nu, target_yplus=target_yplus, rho=rho,
    )
    t_first = yplus_est.first_cell_height

    # δ99 ≈ 0.37 × L / Re_L^(1/5)  (flat plate turbulent)
    Re_L = u_ref * l_ref / nu
    delta99 = 0.37 * l_ref / (Re_L ** 0.2) if Re_L > 0 else 0.01

    # Total thickness alvo: cobrir `coverage_fraction` × δ99
    total_target = coverage_fraction * delta99

    # n_layers tal que t_first × (1 + r + r² + ... + r^(n-1)) ≥ total_target
    # S_n = t_first × (r^n − 1) / (r − 1)
    # n = ln(1 + total*(r-1)/t_first) / ln(r)
    r = expansion_ratio
    if t_first > 0 and r > 1.001:
        n_layers = math.ceil(
            math.log(1 + total_target * (r - 1) / t_first) / math.log(r)
        )
    else:
        n_layers = 10
    n_layers = max(5, min(30, n_layers))

    # Recalcular total exato
    total = t_first * (r ** n_layers - 1) / (r - 1) if r > 1.001 else t_first * n_layers
    final = t_first * r ** (n_layers - 1)

    return PrismLayerConfig(
        n_layers=n_layers,
        first_layer_thickness=t_first,
        expansion_ratio=r,
        total_thickness=total,
        final_layer_thickness=final,
        yplus_target=target_yplus,
        yplus_achieved=yplus_est.y_plus_check,
        delta99_estimate=delta99,
    )


def yplus_target_for_model(turbulence_model: str) -> float:
    """Retornar y+ alvo recomendado para cada modelo de turbulência.

    - k-ε standard + wall functions : y+ ∈ [30, 300]
    - k-ω SST low-Re                : y+ ≈ 1
    - k-ω SST + wall functions      : y+ ∈ [30, 100]
    - γ-Reθ transition              : y+ < 1
    - LES / DES                     : y+ ≈ 1
    """
    m = turbulence_model.lower()
    if "transition" in m or "lm" in m:
        return 0.5
    if "sst" in m or "omega" in m:
        return 1.0
    if "les" in m or "des" in m:
        return 1.0
    return 30.0  # k-ε default with wall functions
