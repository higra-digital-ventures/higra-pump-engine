"""Modelo de transição γ-Reθ (kOmegaSSTLM) — Fase 19.4.

Habilita o modelo de Menter-Langtry γ-Reθ no OpenFOAM (kOmegaSSTLM)
para capturar transição laminar-turbulenta em bombas com Re moderado.

Relevância:
  - Bombas operando em baixa vazão têm boundary layer parcialmente
    laminar na entrada da pá
  - k-ω SST pressupõe turbulência totalmente desenvolvida → superestima
    as perdas de fricção e subestima η
  - γ-Reθ prevê onde ocorre transição → η mais preciso (±1-2%)

Referências:
    - Menter, F.R. & Langtry, R.B. (2009). "A correlation-based
      transition model using local variables" J. Turbomachinery 128.
    - OpenFOAM kOmegaSSTLM turbulenceProperties tutorial.

Usage
-----
    from hpe.cfd.openfoam.transition_model import (
        write_transition_properties, write_gamma_field, write_reTheta_field,
    )

    write_transition_properties(case_dir)
    write_gamma_field(case_dir)
    write_reTheta_field(case_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def write_transition_properties(case_dir: "str | Path") -> None:
    """Sobrescrever turbulenceProperties para kOmegaSSTLM."""
    case_dir = Path(case_dir)
    (case_dir / "constant" / "turbulenceProperties").write_text(
        """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSSTLM;
    turbulence      on;
    printCoeffs     on;

    kOmegaSSTLMCoeffs
    {
        ca1         2.0;
        ca2         0.06;
        ce1         1.0;
        ce2         50.0;
        cThetat     0.03;
        sigmaThetat 2.0;
        lambdaErr   1e-6;
        maxLambdaIter 10;
    }
}
""",
        encoding="utf-8",
    )
    log.info("Wrote kOmegaSSTLM turbulenceProperties at %s", case_dir)


def write_gamma_field(
    case_dir: "str | Path",
    initial_value: float = 1.0,
) -> None:
    """Campo inicial intermittency γ (0 = laminar, 1 = turbulento).

    Por default usamos γ=1 no interior (totalmente turbulento), e deixamos
    o modelo naturalmente determinar onde há laminar/transição.
    Na inlet usa inletOutlet para evitar backflow problems.
    """
    case_dir = Path(case_dir)
    (case_dir / "0" / "gammaInt").write_text(
        f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      gammaInt;
}}

dimensions      [0 0 0 0 0 0 0];
internalField   uniform {initial_value};

boundaryField
{{
    inlet
    {{
        type            inletOutlet;
        inletValue      uniform {initial_value};
        value           uniform {initial_value};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    blade
    {{
        type            zeroGradient;
    }}
    hub
    {{
        type            zeroGradient;
    }}
    shroud
    {{
        type            zeroGradient;
    }}
    "walls.*"
    {{
        type            zeroGradient;
    }}
}}
""",
        encoding="utf-8",
    )


def write_reTheta_field(
    case_dir: "str | Path",
    u_ref: float = 10.0,
    turbulence_intensity: float = 0.05,
) -> None:
    """Campo inicial Reθ_t (Reynolds de transição).

    Correlação de Menter 2009 para Reθ_t em função de Tu (intensidade turbulenta):
        Reθ_t = 1173.51 − 589.428·Tu + 0.2196/Tu²     (Tu ≤ 1.3%)
        Reθ_t = 331.50·(Tu − 0.5658)^(−0.671)          (Tu > 1.3%)
    """
    tu_pct = turbulence_intensity * 100
    if tu_pct <= 1.3:
        re_theta = 1173.51 - 589.428 * tu_pct + 0.2196 / max(tu_pct ** 2, 1e-6)
    else:
        re_theta = 331.50 * (tu_pct - 0.5658) ** (-0.671)
    re_theta = max(20.0, min(2500.0, re_theta))

    case_dir = Path(case_dir)
    (case_dir / "0" / "ReThetat").write_text(
        f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      ReThetat;
}}

dimensions      [0 0 0 0 0 0 0];
internalField   uniform {re_theta:.2f};

boundaryField
{{
    inlet
    {{
        type            inletOutlet;
        inletValue      uniform {re_theta:.2f};
        value           uniform {re_theta:.2f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    "(blade|hub|shroud|walls.*)"
    {{
        type            zeroGradient;
    }}
}}
""",
        encoding="utf-8",
    )
    log.info("Wrote ReThetat = %.1f (Tu = %.2f%%)", re_theta, tu_pct)


def enable_transition_for_case(
    case_dir: "str | Path",
    u_ref: float = 10.0,
    turbulence_intensity: float = 0.05,
) -> dict:
    """Conveniência: ativar γ-Reθ completo em um caso já existente.

    Executa todos os writers e retorna um dict com o resumo.
    """
    case_dir = Path(case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    write_transition_properties(case_dir)
    write_gamma_field(case_dir)
    write_reTheta_field(case_dir, u_ref=u_ref, turbulence_intensity=turbulence_intensity)

    return {
        "case_dir": str(case_dir),
        "model": "kOmegaSSTLM",
        "turbulence_intensity": turbulence_intensity,
        "transition_enabled": True,
        "fields_written": ["gammaInt", "ReThetat"],
    }
