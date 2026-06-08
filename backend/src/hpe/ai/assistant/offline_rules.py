"""Regras baseadas em física para o modo offline do assistente HPE.

Usadas quando Claude API não está disponível. Implementam lógica diagnóstica
baseada em correlações de Gülich (2010) e literatura de turbomáquinas.

Todas as funções retornam dicionários com:
  - 'diagnosis': texto explicativo
  - 'severity': 'ok' | 'warning' | 'critical'
  - 'actions': lista de ações recomendadas
  - 'references': referências bibliográficas relevantes

Referências principais:
  - Gülich, J.F. (2010). Centrifugal Pumps, 2nd ed. Springer.
  - KSB (2012). Selecting Centrifugal Pumps.
  - Hydraulic Institute Standards (HIS 2020).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Constantes físicas e limites operacionais
# ---------------------------------------------------------------------------

_G = 9.80665  # m/s²


# ---------------------------------------------------------------------------
# Análise de cavitação
# ---------------------------------------------------------------------------

def analyze_cavitation_risk(npsh_r: float, npsh_a: float) -> dict:
    """Analisa risco de cavitação e sugere ações corretivas.

    Avalia a margem entre NPSHr (requerido pela bomba) e NPSHa (disponível
    na instalação). Segue critérios do Hydraulic Institute e Gülich §8.2.

    Parâmetros
    ----------
    npsh_r : float
        NPSH requerido pela bomba [m] — determinado pelo fabricante ou sizing 1D.
    npsh_a : float
        NPSH disponível na instalação [m] — calculado pela curva do sistema.

    Retorna
    -------
    dict com:
        diagnosis : str — análise textual
        severity : str — 'ok' | 'warning' | 'critical'
        actions : list[str] — ações recomendadas por prioridade
        margin_m : float — margem NPSHa − NPSHr [m]
        margin_pct : float — margem relativa (%)
        references : list[str]
    """
    margin = npsh_a - npsh_r
    margin_pct = (margin / npsh_r * 100.0) if npsh_r > 0 else 0.0

    actions: list[str] = []
    references = ["Gülich §8.2 — NPSH e cavitação", "HIS 9.6.1 — Margem NPSH"]

    if margin >= 0.5 * npsh_r:
        severity = "ok"
        diagnosis = (
            f"Margem de NPSH adequada: {margin:.2f} m ({margin_pct:.0f}% do NPSHr). "
            "A instalação opera com margem suficiente contra cavitação."
        )
        actions.append("Monitorar NPSHa periodicamente — pode variar com temperatura e altitude.")

    elif margin > 0:
        severity = "warning"
        diagnosis = (
            f"Margem de NPSH baixa: {margin:.2f} m ({margin_pct:.0f}% do NPSHr). "
            "Risco de cavitação em transientes ou variações de temperatura."
        )
        actions.extend([
            "Aumentar NPSHa: elevar nível do reservatório de sucção ou reduzir perdas na tubulação de sucção.",
            "Verificar temperatura do fluido — vapor de água reduz NPSHa.",
            "Considerar impelidor com NPSHr menor (maior olho de sucção).",
        ])
        references.append("Gülich §8.2.4 — Medidas para aumentar NPSHa")

    else:
        severity = "critical"
        diagnosis = (
            f"DEFICIT de NPSH: {abs(margin):.2f} m (NPSHa < NPSHr). "
            "Cavitação severa ocorrerá, causando erosão e queda de desempenho."
        )
        actions.extend([
            "AÇÃO IMEDIATA: Reduzir rotação da bomba para diminuir NPSHr (escala com n²).",
            "Elevar nível do tanque de sucção ou instalar vaso de pressão.",
            "Revisar tubulação de sucção: eliminar singularidades, reduzir comprimento.",
            "Considerar bomba booster na sucção ou substituir por modelo com Nss menor.",
            f"NPSHr necessário: {npsh_r:.1f} m. NPSHa atual: {npsh_a:.1f} m. Deficit: {abs(margin):.1f} m.",
        ])
        references.extend([
            "Gülich §8.4 — Erosão por cavitação",
            "KSB: Selecting Centrifugal Pumps, §3",
        ])

    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "actions": actions,
        "margin_m": round(margin, 3),
        "margin_pct": round(margin_pct, 1),
        "references": references,
    }


# ---------------------------------------------------------------------------
# Diagnóstico de baixa eficiência
# ---------------------------------------------------------------------------

def diagnose_low_efficiency(eta: float, ns: float) -> dict:
    """Diagnostica causas de baixa eficiência por faixa de velocidade específica.

    Compara a eficiência medida com curvas de referência de Gülich (Figura 3.26)
    para identificar possíveis causas e recomendar ações.

    Parâmetros
    ----------
    eta : float
        Eficiência total medida ou predita [fração 0–1].
    ns : float
        Velocidade específica dimensional n·√Q / H^0.75 [rpm, m³/s, m].

    Retorna
    -------
    dict com:
        diagnosis : str
        severity : str
        expected_eta : float — eficiência esperada para este Ns (Gülich)
        deficit_pp : float — déficit em pontos percentuais
        root_causes : list[str]
        actions : list[str]
        references : list[str]
    """
    # Eficiência ótima esperada por faixa de Ns (Gülich Fig. 3.26, vazão ~100 m³/h)
    # Correlação simplificada: eta_opt(Ns) baseada em dados de referência
    if ns < 10:
        eta_ref = 0.55
        regime = "ultra-baixo Ns (<10) — bomba de alto head, baixa vazão"
    elif ns < 20:
        eta_ref = 0.65
        regime = "baixo Ns (10–20) — radial de alto head"
    elif ns < 40:
        eta_ref = 0.78
        regime = "Ns moderado (20–40) — radial padrão"
    elif ns < 70:
        eta_ref = 0.84
        regime = "Ns alto (40–70) — radial de alta eficiência"
    elif ns < 120:
        eta_ref = 0.87
        regime = "misto (70–120) — mixed-flow"
    elif ns < 200:
        eta_ref = 0.88
        regime = "axial moderado (120–200)"
    else:
        eta_ref = 0.85
        regime = "axial alto (>200) — propeller"

    deficit_pp = (eta_ref - eta) * 100.0

    root_causes: list[str] = []
    actions: list[str] = []
    references = ["Gülich §3.10 — Perdas e eficiência", "Gülich Fig. 3.26 — eta_max(Ns)"]

    if deficit_pp <= 0:
        severity = "ok"
        diagnosis = (
            f"Eficiência {eta:.1%} está acima da referência para {regime} "
            f"(esperado: {eta_ref:.1%}). Design competitivo."
        )
    elif deficit_pp <= 5:
        severity = "ok"
        diagnosis = (
            f"Eficiência {eta:.1%} é {deficit_pp:.1f}pp abaixo da referência para {regime} "
            f"(esperado: {eta_ref:.1%}). Dentro da margem aceitável."
        )
        actions.append("Otimização geométrica pode recuperar 1–3pp adicionais.")
    elif deficit_pp <= 12:
        severity = "warning"
        diagnosis = (
            f"Eficiência {eta:.1%} está {deficit_pp:.1f}pp abaixo da referência para {regime} "
            f"(esperado: {eta_ref:.1%}). Investigação recomendada."
        )
        # Causas específicas por regime
        if ns < 30:
            root_causes.extend([
                "Perdas por disco (disk friction) dominantes em baixo Ns",
                "Recirculação na entrada — ponto de operação afastado do BEP",
                "Folga anel de desgaste elevada — aumenta recirculação interna",
            ])
        else:
            root_causes.extend([
                "Geometria de palheta subótima — ângulo beta2 ou b2 inadequado",
                "Rugosidade superficial elevada — impacto maior em alto Re",
                "Operação em carga parcial (<70% do BEP)",
            ])
        actions.extend([
            "Verificar ponto de operação real vs BEP — operar entre 80–115% do BEP.",
            "Inspecionar folgas dos anéis de desgaste — devem ser < 0.3mm por lado.",
            "Avaliar acabamento superficial do canal do impelidor (Ra < 3.2 µm).",
        ])
        references.append("Gülich §3.6 — Perdas no impelidor")

    else:
        severity = "critical"
        diagnosis = (
            f"Eficiência {eta:.1%} está {deficit_pp:.1f}pp abaixo da referência para {regime} "
            f"(esperado: {eta_ref:.1%}). Problema grave — análise detalhada necessária."
        )
        root_causes.extend([
            "Geometria incorreta ou impelidor errado para o ponto de operação",
            "Cavitação severa degradando o escoamento",
            "Recirculação intensa na entrada ou saída",
            "Folga anel de desgaste muito elevada (> 0.5mm)",
            "Impelidor danificado ou com depósito interno",
        ])
        actions.extend([
            "Realizar curva de desempenho completa (Q–H–P) para localizar o problema.",
            "Inspecionar visualmente impelidor e difusor.",
            "Verificar vazão de operação — pode haver bypass não intencional.",
            "Considerar redesign ou substituição do impelidor.",
        ])
        references.extend([
            "Gülich §11 — Problemas de desempenho e diagnóstico",
            "HIS 9.6.4 — Troubleshooting",
        ])

    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "expected_eta": eta_ref,
        "deficit_pp": round(deficit_pp, 1),
        "root_causes": root_causes,
        "actions": actions,
        "references": references,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Sugestões de melhorias geométricas
# ---------------------------------------------------------------------------

def suggest_geometry_improvements(
    d2_mm: float,
    b2_mm: float,
    beta2: float,
    ns: float,
) -> list[str]:
    """Sugere melhorias geométricas baseadas em correlações de Gülich §3.7.

    Avalia os parâmetros geométricos do impelidor e propõe ajustes para
    maximizar eficiência na faixa de velocidade específica.

    Parâmetros
    ----------
    d2_mm : float
        Diâmetro externo do impelidor [mm].
    b2_mm : float
        Largura da saída do impelidor [mm].
    beta2 : float
        Ângulo da palheta na saída [graus].
    ns : float
        Velocidade específica dimensional.

    Retorna
    -------
    list[str]
        Lista de sugestões de melhoria, ordenadas por impacto estimado.
    """
    suggestions: list[str] = []

    # Relação b2/D2 ótima (Gülich Tab. 3.2)
    b2_d2 = b2_mm / d2_mm if d2_mm > 0 else 0.05
    b2_d2_opt_low, b2_d2_opt_high = _b2_d2_optimal_range(ns)

    if b2_d2 < b2_d2_opt_low:
        deficit_pct = (b2_d2_opt_low - b2_d2) / b2_d2_opt_low * 100
        suggestions.append(
            f"Aumentar largura b2: relação b2/D2={b2_d2:.3f} está {deficit_pct:.0f}% abaixo "
            f"do ótimo [{b2_d2_opt_low:.3f}–{b2_d2_opt_high:.3f}] para Ns={ns:.0f}. "
            "Largura insuficiente aumenta velocidade meridional e perdas por atrito. "
            f"Sugestão: b2 ≥ {b2_d2_opt_low * d2_mm:.0f} mm. (Gülich Tab. 3.2)"
        )
    elif b2_d2 > b2_d2_opt_high:
        excess_pct = (b2_d2 - b2_d2_opt_high) / b2_d2_opt_high * 100
        suggestions.append(
            f"Reduzir largura b2: relação b2/D2={b2_d2:.3f} está {excess_pct:.0f}% acima "
            f"do ótimo [{b2_d2_opt_low:.3f}–{b2_d2_opt_high:.3f}] para Ns={ns:.0f}. "
            "Passagem larga favorece recirculação na saída. "
            f"Sugestão: b2 ≤ {b2_d2_opt_high * d2_mm:.0f} mm. (Gülich §3.7.1)"
        )

    # Ângulo beta2 ótimo (Gülich §3.3, Tab. 3.3)
    beta2_opt_low, beta2_opt_high = _beta2_optimal_range(ns)
    if beta2 < beta2_opt_low:
        suggestions.append(
            f"Aumentar ângulo beta2: {beta2:.1f}° está abaixo da faixa ótima "
            f"[{beta2_opt_low:.0f}°–{beta2_opt_high:.0f}°] para Ns={ns:.0f}. "
            "Beta2 muito baixo reduz coeficiente de head e eficiência. "
            "(Gülich §3.3.4 — Velocidade específica e ângulo de palheta)"
        )
    elif beta2 > beta2_opt_high:
        suggestions.append(
            f"Reduzir ângulo beta2: {beta2:.1f}° está acima da faixa ótima "
            f"[{beta2_opt_low:.0f}°–{beta2_opt_high:.0f}°] para Ns={ns:.0f}. "
            "Beta2 excessivo aumenta instabilidade e risco de curva H-Q não monotônica. "
            "(Gülich §5.2 — Instabilidade da curva)"
        )

    # Velocidade periférica e desgaste (função de d2 e rotação estimada)
    # Para verificação: u2 = π*D2*n/60. Sem 'n', estimamos por Ns.
    # Ns = n*√Q / H^0.75 → não podemos calcular u2 sem Q e H
    # Então usamos d2_mm como proxy de tamanho e Ns para estimar u2 relativo
    if d2_mm > 500 and ns < 30:
        suggestions.append(
            f"Diâmetro D2={d2_mm:.0f} mm é grande para Ns={ns:.0f}. "
            "Em bombas de baixo Ns, grandes diâmetros aumentam perdas por disco (disk friction). "
            "Considerar multiestagio com D2 menor. (Gülich §3.6.4)"
        )

    # Número de palhetas (heurística Gülich §3.7.3)
    # z_opt ≈ 6.5*(1 + D1/D2)*sin(beta1_med + beta2)/2
    # Sem D1 e beta1, usamos correlação simplificada com Ns
    z_opt_min, z_opt_max = _z_optimal_range(ns)
    suggestions.append(
        f"Número de palhetas recomendado para Ns={ns:.0f}: {z_opt_min}–{z_opt_max} palhetas. "
        f"Verificar se o design atual está nesta faixa. (Gülich §3.7.3)"
    )

    # Acabamento superficial (impacto em eficiência)
    if ns > 60:
        suggestions.append(
            "Para Ns > 60, acabamento superficial tem impacto significativo. "
            "Rugosidade Ra < 1.6 µm no canal do impelidor pode recuperar 1–2pp de eficiência. "
            "(Gülich §3.10.3 — Efeito de rugosidade)"
        )

    return suggestions


def _b2_d2_optimal_range(ns: float) -> tuple[float, float]:
    """Faixa ótima de b2/D2 por velocidade específica (Gülich Tab. 3.2)."""
    if ns < 15:
        return 0.02, 0.05
    elif ns < 30:
        return 0.04, 0.08
    elif ns < 50:
        return 0.06, 0.12
    elif ns < 80:
        return 0.09, 0.16
    elif ns < 120:
        return 0.12, 0.22
    else:
        return 0.18, 0.35


def _beta2_optimal_range(ns: float) -> tuple[float, float]:
    """Faixa ótima de ângulo beta2 por velocidade específica (Gülich §3.3)."""
    if ns < 20:
        return 15.0, 25.0
    elif ns < 40:
        return 20.0, 30.0
    elif ns < 70:
        return 22.0, 35.0
    elif ns < 120:
        return 25.0, 40.0
    else:
        return 30.0, 50.0


def _z_optimal_range(ns: float) -> tuple[int, int]:
    """Faixa ótima de número de palhetas por velocidade específica (Gülich §3.7.3)."""
    if ns < 20:
        return 7, 9
    elif ns < 40:
        return 6, 8
    elif ns < 70:
        return 5, 7
    elif ns < 120:
        return 4, 6
    else:
        return 3, 5


# ---------------------------------------------------------------------------
# Explicação de bias sistemático
# ---------------------------------------------------------------------------

def explain_bias(bias_pp: float, pump_type: str = "radial") -> str:
    """Explica o bias sistemático entre cálculo 1D e bancada.

    O sizing 1D (correlações de Gülich) assume geometria ótima, enquanto
    os dados de bancada refletem rotores reais com trimagem, acabamento
    industrial e tolerâncias de fabricação.

    Parâmetros
    ----------
    bias_pp : float
        Bias em pontos percentuais: positivo = cálculo superestima.
        Ex: bias_pp=+7.65 significa sizing prediz 7.65pp acima da bancada.
    pump_type : str
        Tipo de bomba: 'radial', 'mixed', 'axial'.

    Retorna
    -------
    str
        Explicação textual do bias com causas e implicações.
    """
    abs_bias = abs(bias_pp)
    direction = "superestima" if bias_pp > 0 else "subestima"

    causes_radial = [
        "Trimagem do rotor: impelidor usinado para ajuste de curva reduz eficiência ~1–3pp vs design ótimo.",
        "Acabamento industrial: correlações assumem superfície lisa; rotores reais têm Ra > 3.2 µm.",
        "Folgas reais: anéis de desgaste com folgas de fabricação maiores que o design teórico.",
        "Pré-rotação na entrada: instalações reais raramente têm escoamento perfeitamente axial.",
        "Escalonamento geométrico: correlações calibradas para tamanhos médios; bombas pequenas/grandes desviam.",
    ]

    causes_mixed = causes_radial + [
        "Interação rotor-estator: bombas mistas são mais sensíveis a folgas radiais.",
    ]

    causes_axial = [
        "Folga de ponta (tip clearance): impacto muito maior em axiais (perda ~0.5pp por 0.1mm de folga).",
        "Distribuição de carga na palheta: correlações 1D não capturam gradientes 3D.",
    ]

    causes = causes_axial if pump_type == "axial" else (causes_mixed if pump_type == "mixed" else causes_radial)

    if abs_bias <= 5:
        assessment = "Excelente concordância. Bias dentro da incerteza das correlações de Gülich."
        action = "Nenhuma ação corretiva necessária. Aplicar correção pontual de -7pp para análise conservadora."
    elif abs_bias <= 15:
        assessment = (
            f"Bias de {bias_pp:+.1f}pp é típico para sizing 1D vs dados industriais. "
            "A validação M1.8 da HPE (MAPE 11.69%) confirma performance dentro dos critérios."
        )
        action = (
            f"Para estimativas de campo, aplicar fator de correção de {-bias_pp:+.1f}pp. "
            "Calibrar correlação com dados específicos da família de bombas em questão."
        )
    else:
        assessment = (
            f"Bias de {bias_pp:+.1f}pp é elevado — indica que as correlações precisam de recalibração "
            "para esta faixa específica de Ns, D2 ou condição de operação."
        )
        action = (
            "Revisar correlações de eficiência para a faixa de Ns em questão. "
            "Coletar mais dados de bancada para recalibrar o modelo. "
            "Verificar se a condição de operação está próxima ao BEP."
        )

    lines = [
        f"Bias 1D vs bancada: cálculo {direction} em {abs_bias:.1f} pontos percentuais.",
        "",
        f"Avaliação: {assessment}",
        "",
        "Principais causas do bias sistemático:",
    ]
    for i, cause in enumerate(causes, 1):
        lines.append(f"  {i}. {cause}")

    lines.extend([
        "",
        f"Recomendação: {action}",
        "",
        "Referência: Gülich §3.10.6 — Incerteza das correlações de eficiência.",
        "Nota HPE: Validação M1.8 (435 pontos) mostrou bias médio de +7.65pp — esperado para design ótimo vs rotor trimado.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Análise de triângulos de velocidade
# ---------------------------------------------------------------------------

def analyze_velocity_triangles(
    u1: float,
    u2: float,
    cm1: float,
    cm2: float,
    beta1: float,
    beta2: float,
    cu1: float = 0.0,
) -> dict:
    """Analisa os triângulos de velocidade para consistência física.

    Verifica conformidade com critérios de Gülich §3.2–§3.3.

    Parâmetros
    ----------
    u1, u2 : float — velocidades periféricas entrada/saída [m/s]
    cm1, cm2 : float — velocidades meridionais entrada/saída [m/s]
    beta1, beta2 : float — ângulos de palheta entrada/saída [graus]
    cu1 : float — componente tangencial na entrada [m/s] (default 0)

    Retorna
    -------
    dict com análise dos triângulos.
    """
    import math

    warnings: list[str] = []
    info: list[str] = []

    # Velocidade relativa na entrada
    wu1 = u1 - cu1
    w1 = math.sqrt(cm1**2 + wu1**2) if cm1 > 0 else 0.0
    beta1_calc = math.degrees(math.atan2(cm1, wu1)) if wu1 > 0 else 90.0

    # Velocidade relativa na saída
    beta2_rad = math.radians(beta2)
    cu2 = u2 - cm2 / math.tan(beta2_rad) if beta2 != 90 else u2
    w2 = math.sqrt(cm2**2 + (u2 - cu2)**2)
    c2 = math.sqrt(cm2**2 + cu2**2)

    # Cabeça de Euler
    g = 9.80665
    H_euler = (u2 * cu2 - u1 * cu1) / g

    # Razão de desaceleração w1/w2 (Gülich §3.3.2)
    decel_ratio = w1 / w2 if w2 > 0 else 0.0
    if decel_ratio > 1.4:
        warnings.append(
            f"Razão de desaceleração w1/w2={decel_ratio:.2f} > 1.4 — risco de separação do escoamento. "
            "Aumentar número de palhetas ou ajustar geometria do canal. (Gülich §3.3.2)"
        )
    elif decel_ratio > 1.2:
        info.append(f"Razão w1/w2={decel_ratio:.2f} aceitável (< 1.4).")
    else:
        info.append(f"Razão w1/w2={decel_ratio:.2f} excelente (< 1.2 — escoamento bem guiado).")

    # Ângulo de incidência real vs geometria
    incidence = beta1_calc - beta1
    if abs(incidence) > 5.0:
        warnings.append(
            f"Incidência na entrada: {incidence:+.1f}° (beta1_calc={beta1_calc:.1f}°, "
            f"beta1_geo={beta1:.1f}°). Incidência > ±5° aumenta perdas de choque. "
            "(Gülich §5.2.2)"
        )
    else:
        info.append(f"Incidência na entrada: {incidence:+.1f}° — dentro da faixa aceitável.")

    # Velocidade periférica u2 (limite de erosão/ruído)
    if u2 > 50:
        warnings.append(
            f"Velocidade periférica u2={u2:.1f} m/s > 50 m/s. "
            "Risco de ruído aerodinâmico e erosão do anel de desgaste. "
            "(Gülich §10.1 — Ruído em altas velocidades)"
        )
    elif u2 > 40:
        info.append(f"u2={u2:.1f} m/s — monitorar ruído e erosão acima de 50 m/s.")

    return {
        "H_euler_m": round(H_euler, 2),
        "cu2_m_s": round(cu2, 3),
        "w1_m_s": round(w1, 3),
        "w2_m_s": round(w2, 3),
        "decel_ratio": round(decel_ratio, 3),
        "incidence_deg": round(incidence, 2),
        "warnings": warnings,
        "info": info,
        "severity": "warning" if warnings else "ok",
    }
