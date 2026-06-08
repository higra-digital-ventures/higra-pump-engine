"""RAG Engineering Assistant para HPE.

Recuperação Aumentada por Geração (RAG) adaptada para turbomáquinas.
Usa documentos de engenharia embutidos (sem servidor externo necessário).

Modos:
  1. Offline (padrão): regras determinísticas de offline_rules.py + knowledge base
  2. Online (opcional): chama Claude API quando ANTHROPIC_API_KEY disponível

Fonte de conhecimento (embutida):
  - Correlações de Gülich (2010), capítulos 1-8
  - Critérios de design ADT/HIGRA
  - Padrões ISO/HIS

Uso:
    assistant = EngineeringAssistant()
    response = assistant.ask("Minha bomba tem eficiência baixa. D2=300mm, Ns=35. O que verificar?")

    # Ou com contexto de sizing:
    sizing = run_sizing(op)
    response = assistant.diagnose(sizing, question="Por que o NPSHr está alto?")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from hpe.ai.assistant.offline_rules import (
    analyze_cavitation_risk,
    diagnose_low_efficiency,
    suggest_geometry_improvements,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base de conhecimento embutida (fragmentos textuais para RAG)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "gulich_nq_radial",
        "topic": "specific_speed",
        "keywords": ["ns", "nq", "radial", "velocidade especifica", "head", "alto head"],
        "text": (
            "Nq < 25: bomba radial de alto head. D2 grande, b2/D2 pequeno (~0.02-0.04). "
            "Risco: eficiência reduzida por atrito de disco. Recomendar: verificar D1/D2 ratio."
        ),
        "references": ["Gülich 2010, §2.1"],
    },
    {
        "id": "gulich_nq_mixed",
        "topic": "specific_speed",
        "keywords": ["ns", "nq", "misto", "mixed", "diagonal", "beta2", "60", "120"],
        "text": (
            "Nq 60-120: bomba mista. Impelidor diagonal. Maior coeficiente de vazão. "
            "Beta2 típico 18-30°. NPSHr mais sensível à rotação."
        ),
        "references": ["Gülich 2010, §2.2"],
    },
    {
        "id": "gulich_cavitation",
        "topic": "cavitation",
        "keywords": [
            "cavitacao", "cavitação", "npsh", "npshr", "npsha", "sigma", "sucção",
            "alto", "reduzir", "indutor", "erosão",
        ],
        "text": (
            "NPSHr mínimo ocorre a Q≈1.1*Q_bep. Sigma de Thoma: σ = NPSHr/H. "
            "Para sigma < 0.05: bomba resistente à cavitação. "
            "Medidas para reduzir NPSHr: aumentar D1, reduzir n, usar indutor."
        ),
        "references": ["Gülich 2010, §6.1", "HIS 9.6.6"],
    },
    {
        "id": "gulich_efficiency",
        "topic": "efficiency",
        "keywords": [
            "eficiencia", "eficiência", "eta", "baixa", "rendimento",
            "perda", "hidraulica", "volumetrica", "mecanica",
        ],
        "text": (
            "Eficiência total: η = η_hid * η_vol * η_mec. "
            "Perdas típicas: η_hid≈0.92-0.96, η_vol≈0.96-0.99, η_mec≈0.97-0.99. "
            "Abaixo de 70%: verificar folgas de desgaste, rugosidade superficial, recirculação."
        ),
        "references": ["Gülich 2010, §3.9"],
    },
    {
        "id": "gulich_u2_limit",
        "topic": "materials",
        "keywords": [
            "u2", "velocidade periferica", "erosão", "erosao", "material",
            "aco", "aço", "inox", "abrasivo", "desgaste",
        ],
        "text": (
            "Velocidade de ponta u2 > 35 m/s: risco de erosão em aço fundido. "
            "Para água limpa: u2 até 45 m/s com aço inox. "
            "Para fluidos abrasivos: limitar u2 < 25 m/s."
        ),
        "references": ["Gülich 2010, §14.3"],
    },
    {
        "id": "gulich_beta2",
        "topic": "blade_angle",
        "keywords": [
            "beta2", "angulo", "ângulo", "palheta", "saida", "instabilidade",
            "escorregamento", "curva", "otimo",
        ],
        "text": (
            "Beta2 ótimo: 20-28° para η máxima. Beta2 < 18°: curva H-Q instável. "
            "Beta2 > 35°: H/Hth < 0.85 (escorregamento excessivo)."
        ),
        "references": ["Gülich 2010, §3.3"],
    },
    {
        "id": "gulich_stall",
        "topic": "stability",
        "keywords": [
            "instabilidade", "surge", "estabilidade", "curva", "hq", "dh", "dq",
            "recirculacao", "recirculação", "parcial",
        ],
        "text": (
            "dH/dQ > 0 na curva H-Q: instabilidade (surge). Causa: Nq muito baixo + beta2 grande. "
            "Solução: aumentar Nq (aumentar n ou reduzir H), reduzir beta2."
        ),
        "references": ["Gülich 2010, §5.4"],
    },
    {
        "id": "gulich_wear_ring",
        "topic": "wear",
        "keywords": [
            "anel", "desgaste", "folga", "volumetrica", "vazão cai", "q cai",
            "wear ring", "clearance",
        ],
        "text": (
            "Folga de anel de desgaste: s = 0.15 + 0.002*D [mm]. "
            "Folga excessiva → η_vol < 0.95. Sintoma: Q cai sem mudar pressão."
        ),
        "references": ["Gülich 2010, §9.1"],
    },
    {
        "id": "gulich_bep",
        "topic": "operating_point",
        "keywords": [
            "bep", "ponto ótimo", "melhor ponto", "ponto de operacao",
            "operacao parcial", "sobrecarga", "carga parcial",
        ],
        "text": (
            "BEP (Best Efficiency Point): operar entre 80-115% do BEP garante eficiência máxima "
            "e vida útil longa. Fora desta faixa: recirculação, cavitação e vibração aumentam. "
            "Operação contínua < 70% BEP: não recomendada."
        ),
        "references": ["Gülich 2010, §5.1", "HIS 9.6.3"],
    },
    {
        "id": "gulich_multistage",
        "topic": "multistage",
        "keywords": [
            "multiestágio", "multiestagio", "estagios", "estágios", "stages",
            "alto head", "serie", "ns baixo",
        ],
        "text": (
            "Multiestagio indicado quando Ns < 15 por estágio. "
            "Vantagens: maior eficiência, D2 menor, melhor controle de u2. "
            "Desvantagem: custo e complexidade de montagem."
        ),
        "references": ["Gülich 2010, §2.5"],
    },
    {
        "id": "gulich_surface_finish",
        "topic": "surface_finish",
        "keywords": [
            "rugosidade", "acabamento", "ra", "superficie", "superfície",
            "polimento", "usinagem", "eficiencia", "recuperar",
        ],
        "text": (
            "Rugosidade Ra < 1.6 µm no canal do impelidor pode recuperar 1-2pp de eficiência. "
            "Impacto maior em Ns > 60 e grandes diâmetros. "
            "Carcaça: Ra < 3.2 µm. Polimento eletrolítico: Ra < 0.8 µm."
        ),
        "references": ["Gülich 2010, §3.10.3"],
    },
    {
        "id": "gulich_d1_suction",
        "topic": "suction_design",
        "keywords": [
            "d1", "olho de sucção", "sucção", "diametro entrada", "nss",
            "suction", "inlet", "npsh reduzir",
        ],
        "text": (
            "Olho de sucção ótimo: D1/D2 ≈ 0.45-0.60 para Ns 20-60. "
            "D1 maior → NPSHr menor, mas aumenta recirculação de entrada. "
            "Número de sucção específico: Nss = n·√Q / NPSHr^0.75. Manter Nss < 220 (SI)."
        ),
        "references": ["Gülich 2010, §6.2", "HIS 9.6.7"],
    },
]


# ---------------------------------------------------------------------------
# Resposta do assistente
# ---------------------------------------------------------------------------

@dataclass
class AssistantResponse:
    """Resposta estruturada do EngineeringAssistant.

    Atributos
    ----------
    question : str
        Pergunta original.
    answer : str
        Resposta gerada (texto).
    relevant_topics : list[str]
        Tópicos identificados na pergunta.
    recommendations : list[str]
        Lista de ações/recomendações extraídas.
    references : list[str]
        Referências bibliográficas citadas.
    confidence : float
        Score de confiança 0-1 (1 = alta relevância dos fragmentos recuperados).
    mode : str
        Modo de geração: 'offline_rules' | 'rag_local' | 'claude_api'.
    """
    question: str
    answer: str
    relevant_topics: list[str]
    recommendations: list[str]
    references: list[str]
    confidence: float
    mode: str


# ---------------------------------------------------------------------------
# EngineeringAssistant
# ---------------------------------------------------------------------------

class EngineeringAssistant:
    """Assistente de engenharia hidráulica para HPE.

    Responde perguntas sobre design de bombas centrífugas usando:
    1. Base de conhecimento local (KNOWLEDGE_BASE + offline_rules)
    2. Claude API (se ANTHROPIC_API_KEY disponível e use_claude_api=True)

    Parâmetros
    ----------
    use_claude_api : bool
        Se True, tenta usar a API Claude quando ANTHROPIC_API_KEY estiver disponível.
        Default: False (modo offline puro).

    Exemplos
    --------
    >>> assistant = EngineeringAssistant()
    >>> r = assistant.ask("NPSHr alto, como reduzir?")
    >>> print(r.answer)

    >>> r = assistant.ask("Eficiência de 68% para Ns=35. Normal?", context={"ns": 35, "eta": 0.68})
    >>> print(r.recommendations)
    """

    def __init__(self, use_claude_api: bool = False) -> None:
        self.use_claude_api = use_claude_api and bool(os.getenv("ANTHROPIC_API_KEY"))
        log.debug(
            "EngineeringAssistant inicializado (mode=%s)",
            "claude_api" if self.use_claude_api else "offline",
        )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        context: Optional[dict] = None,
    ) -> AssistantResponse:
        """Responde uma pergunta de engenharia de bombas.

        Parâmetros
        ----------
        question : str
            Pergunta em linguagem natural sobre design, diagnóstico ou operação.
        context : dict | None
            Contexto opcional com parâmetros conhecidos:
            - 'ns': velocidade específica
            - 'eta': eficiência total [fração]
            - 'npsh_r': NPSHr [m]
            - 'npsh_a': NPSHa [m]
            - 'd2_mm': diâmetro externo [mm]
            - 'b2_mm': largura saída [mm]
            - 'beta2': ângulo palheta saída [graus]

        Retorna
        -------
        AssistantResponse
        """
        # 1. Recuperar fragmentos relevantes
        retrieved = self._retrieve(question, n=3)

        # 2. Extrair tópicos relevantes
        relevant_topics = list({doc["topic"] for doc in retrieved})

        # 3. Aplicar regras offline
        offline_insights = self._apply_offline_rules(question, context)

        # 4. Gerar resposta
        if self.use_claude_api:
            try:
                answer = self._generate_with_claude(question, retrieved, context)
                mode = "claude_api"
            except Exception as exc:
                log.warning("Claude API falhou (%s) — usando modo offline", exc)
                answer = self._generate_offline(question, retrieved, context, offline_insights)
                mode = "rag_local"
        else:
            answer = self._generate_offline(question, retrieved, context, offline_insights)
            mode = "rag_local" if retrieved else "offline_rules"

        # 5. Extrair recomendações e referências dos fragmentos
        recommendations = self._extract_recommendations(retrieved, offline_insights)
        references = []
        for doc in retrieved:
            references.extend(doc.get("references", []))

        # 6. Score de confiança baseado em sobreposição de palavras-chave
        confidence = self._compute_confidence(question, retrieved)

        return AssistantResponse(
            question=question,
            answer=answer,
            relevant_topics=relevant_topics,
            recommendations=recommendations,
            references=list(dict.fromkeys(references)),  # deduplica, mantém ordem
            confidence=confidence,
            mode=mode,
        )

    def diagnose(
        self,
        sizing_result,
        question: str = "",
    ) -> AssistantResponse:
        """Diagnostica resultado de sizing e responde pergunta específica.

        Parâmetros
        ----------
        sizing_result : objeto ou dict
            Resultado de sizing com atributos como eta_total, npsh_r, ns, d2_mm, etc.
            Pode ser um dataclass ou um dict.
        question : str
            Pergunta adicional sobre o resultado.

        Retorna
        -------
        AssistantResponse
        """
        # Extrair parâmetros do resultado de sizing
        context = _extract_sizing_context(sizing_result)

        # Montar pergunta enriquecida
        full_question = question or "Diagnostique o design desta bomba."
        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items() if v is not None)
            full_question = f"{full_question} [Contexto: {ctx_str}]"

        return self.ask(full_question, context=context)

    # ------------------------------------------------------------------
    # Recuperação (TF-IDF simplificado por palavras-chave)
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, n: int = 3) -> list[dict]:
        """Busca os N fragmentos mais relevantes da base de conhecimento.

        Usa matching simples: conta quantas palavras-chave do fragmento
        aparecem na pergunta (tokenização por espaço/pontuação, case-insensitive).

        Parâmetros
        ----------
        question : str
            Pergunta do usuário.
        n : int
            Número de fragmentos a retornar.

        Retorna
        -------
        list[dict]
            Lista de fragmentos ordenados por relevância (mais relevante primeiro).
        """
        question_lower = question.lower()
        # Tokenizar pergunta (remover pontuação básica)
        for ch in ".,;:!?()[]{}\"'":
            question_lower = question_lower.replace(ch, " ")
        question_tokens = set(question_lower.split())

        scores: list[tuple[float, dict]] = []
        for doc in KNOWLEDGE_BASE:
            # Contar keywords do fragmento que aparecem na pergunta
            keyword_hits = sum(
                1 for kw in doc.get("keywords", [])
                if kw.lower() in question_lower
            )
            # Bonus: palavras do tópico na pergunta
            topic_hit = 1 if doc["topic"].replace("_", " ") in question_lower else 0
            # Bonus: palavras do texto do fragmento
            text_tokens = set(doc["text"].lower().split())
            text_overlap = len(question_tokens & text_tokens) / max(1, len(question_tokens))

            score = keyword_hits * 2.0 + topic_hit * 1.5 + text_overlap * 0.5
            scores.append((score, doc))

        # Ordenar por score decrescente
        scores.sort(key=lambda x: x[0], reverse=True)

        # Retornar apenas fragmentos com score > 0, ou os top-N mesmo assim
        top = [doc for score, doc in scores[:n] if score > 0]
        if not top and scores:
            # Nenhum match — retornar o fragmento mais genérico (eficiência)
            top = [scores[0][1]]

        return top

    # ------------------------------------------------------------------
    # Regras offline
    # ------------------------------------------------------------------

    def _apply_offline_rules(
        self,
        question: str,
        context: Optional[dict],
    ) -> list[dict]:
        """Aplica regras diagnósticas baseadas em parâmetros do contexto.

        Retorna lista de resultados de diagnóstico das regras offline.
        """
        results: list[dict] = []
        if context is None:
            return results

        question_lower = question.lower()

        # Análise de cavitação
        npsh_r = context.get("npsh_r") or context.get("npsh_r_m")
        npsh_a = context.get("npsh_a")
        if npsh_r and npsh_a:
            try:
                r = analyze_cavitation_risk(npsh_r, npsh_a)
                results.append({"type": "cavitation", **r})
            except Exception as exc:
                log.debug("analyze_cavitation_risk falhou: %s", exc)

        # Diagnóstico de eficiência
        eta = context.get("eta") or context.get("eta_total")
        ns = context.get("ns") or context.get("feat_ns")
        if eta is not None and ns is not None:
            # Converter fração para decimal se necessário
            eta_frac = eta / 100.0 if eta > 1.5 else eta
            try:
                r = diagnose_low_efficiency(eta_frac, float(ns))
                results.append({"type": "efficiency", **r})
            except Exception as exc:
                log.debug("diagnose_low_efficiency falhou: %s", exc)

        # Sugestões geométricas
        d2_mm = context.get("d2_mm")
        b2_mm = context.get("b2_mm")
        beta2 = context.get("beta2") or context.get("beta2_deg")
        if d2_mm and b2_mm and beta2 and ns:
            try:
                suggestions = suggest_geometry_improvements(
                    float(d2_mm), float(b2_mm), float(beta2), float(ns)
                )
                if suggestions:
                    results.append({
                        "type": "geometry",
                        "suggestions": suggestions,
                        "severity": "info",
                    })
            except Exception as exc:
                log.debug("suggest_geometry_improvements falhou: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Geração offline
    # ------------------------------------------------------------------

    def _generate_offline(
        self,
        question: str,
        retrieved: list[dict],
        context: Optional[dict],
        offline_insights: Optional[list[dict]] = None,
    ) -> str:
        """Gera resposta usando regras locais + fragmentos recuperados.

        Formato da resposta:
          1. Análise diagnóstica (se houver dados)
          2. Conhecimento de referência (Gülich)
          3. Recomendações

        Parâmetros
        ----------
        question : str
        retrieved : list[dict] — fragmentos da base de conhecimento
        context : dict | None
        offline_insights : list[dict] | None — resultados de offline_rules

        Retorna
        -------
        str — resposta em português
        """
        parts: list[str] = []

        # ---- Diagnóstico baseado em parâmetros (offline rules) ----
        if offline_insights:
            parts.append("## Diagnóstico")
            for insight in offline_insights:
                itype = insight.get("type", "")
                if itype == "cavitation":
                    sev = insight.get("severity", "ok")
                    parts.append(f"**Cavitação ({sev.upper()}):** {insight.get('diagnosis', '')}")
                    actions = insight.get("actions", [])
                    if actions:
                        parts.append("Ações recomendadas:")
                        for a in actions[:3]:
                            parts.append(f"  - {a}")

                elif itype == "efficiency":
                    sev = insight.get("severity", "ok")
                    parts.append(f"**Eficiência ({sev.upper()}):** {insight.get('diagnosis', '')}")
                    root_causes = insight.get("root_causes", [])
                    if root_causes:
                        parts.append("Possíveis causas:")
                        for c in root_causes[:3]:
                            parts.append(f"  - {c}")
                    actions = insight.get("actions", [])
                    if actions:
                        parts.append("Ações:")
                        for a in actions[:2]:
                            parts.append(f"  - {a}")

                elif itype == "geometry":
                    parts.append("**Sugestões geométricas:**")
                    for s in insight.get("suggestions", [])[:2]:
                        # Truncar sugestão longa
                        s_short = s[:200] + "..." if len(s) > 200 else s
                        parts.append(f"  - {s_short}")

        # ---- Conhecimento de referência (fragmentos RAG) ----
        if retrieved:
            parts.append("\n## Base de Conhecimento (Gülich)")
            for doc in retrieved:
                parts.append(f"**{doc['topic'].replace('_', ' ').title()}:** {doc['text']}")

        # ---- Contexto adicional ----
        if context:
            context_items = {
                k: v for k, v in context.items()
                if v is not None and k in (
                    "ns", "eta", "eta_total", "npsh_r", "npsh_a",
                    "d2_mm", "b2_mm", "beta2", "u2"
                )
            }
            if context_items:
                parts.append("\n## Parâmetros Recebidos")
                for k, v in context_items.items():
                    if isinstance(v, float):
                        parts.append(f"  - {k}: {v:.3g}")
                    else:
                        parts.append(f"  - {k}: {v}")

        # ---- Fallback: resposta genérica ----
        if not parts:
            parts.append(
                "Não foram encontrados dados suficientes para um diagnóstico preciso. "
                "Forneça parâmetros como Ns, eficiência (eta), D2, NPSHr/NPSHa para "
                "uma análise mais detalhada."
            )
            parts.append(
                "\nPara questões sobre design de bombas centrífugas, consulte: "
                "Gülich (2010) Centrifugal Pumps, Hydraulic Institute Standards."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Geração com Claude API
    # ------------------------------------------------------------------

    def _generate_with_claude(
        self,
        question: str,
        retrieved: list[dict],
        context: Optional[dict],
    ) -> str:
        """Gera resposta usando Claude API com contexto de engenharia.

        Constrói um prompt com os fragmentos recuperados e contexto do problema,
        então chama o modelo claude-3-haiku para gerar resposta concisa.

        Parâmetros
        ----------
        question, retrieved, context : ver ask()

        Retorna
        -------
        str — resposta gerada pelo modelo
        """
        import anthropic  # type: ignore[import]

        client = anthropic.Anthropic()

        # Construir contexto de engenharia
        knowledge_text = "\n".join(
            f"[{doc['topic']}] {doc['text']} (Ref: {', '.join(doc['references'])})"
            for doc in retrieved
        )

        context_text = ""
        if context:
            ctx_items = {k: v for k, v in context.items() if v is not None}
            if ctx_items:
                context_text = "Parâmetros conhecidos: " + ", ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in ctx_items.items()
                )

        system_prompt = (
            "Você é um especialista em design de bombas centrífugas com profundo conhecimento "
            "de Gülich (2010) Centrifugal Pumps e normas HIS/ISO. "
            "Responda de forma técnica, concisa e em português. "
            "Use os fragmentos de conhecimento fornecidos como base principal. "
            "Cite referências específicas de Gülich quando relevante."
        )

        user_prompt = (
            f"Fragmentos de conhecimento relevantes:\n{knowledge_text}\n\n"
            f"{context_text}\n\n"
            f"Pergunta: {question}"
        )

        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return message.content[0].text

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _extract_recommendations(
        self,
        retrieved: list[dict],
        offline_insights: list[dict],
    ) -> list[str]:
        """Extrai lista unificada de recomendações das regras offline e fragmentos.

        Parâmetros
        ----------
        retrieved : fragmentos da base de conhecimento
        offline_insights : diagnósticos das offline_rules

        Retorna
        -------
        list[str] — recomendações únicas, ordenadas por prioridade
        """
        recs: list[str] = []

        # Ações das regras offline (alta prioridade)
        for insight in offline_insights:
            recs.extend(insight.get("actions", []))
            recs.extend(insight.get("suggestions", []))

        # Extrair recomendações implícitas dos fragmentos (frases com verbos de ação)
        action_verbs = [
            "verificar", "aumentar", "reduzir", "considerar", "inspecionar",
            "limitar", "manter", "recomendar", "avaliar",
        ]
        for doc in retrieved:
            for sentence in doc["text"].split(". "):
                sentence_lower = sentence.lower()
                if any(v in sentence_lower for v in action_verbs):
                    recs.append(sentence.strip())

        # Deduplica mantendo ordem
        seen: set[str] = set()
        unique_recs: list[str] = []
        for r in recs:
            if r and r not in seen:
                seen.add(r)
                unique_recs.append(r)

        return unique_recs[:8]  # Limitar a 8 recomendações

    def _compute_confidence(
        self,
        question: str,
        retrieved: list[dict],
    ) -> float:
        """Score de confiança 0-1 baseado na sobreposição de palavras-chave.

        Parâmetros
        ----------
        question : str
        retrieved : list[dict]

        Retorna
        -------
        float — 0 (sem relevância) a 1 (alta relevância)
        """
        if not retrieved:
            return 0.1

        question_lower = question.lower()
        total_hits = 0
        total_keywords = 0

        for doc in retrieved:
            keywords = doc.get("keywords", [])
            total_keywords += len(keywords)
            hits = sum(1 for kw in keywords if kw.lower() in question_lower)
            total_hits += hits

        if total_keywords == 0:
            return 0.3

        raw_score = total_hits / total_keywords
        # Escalar para [0.2, 0.95]
        return round(min(0.95, 0.2 + raw_score * 0.75), 3)


# ---------------------------------------------------------------------------
# Função auxiliar para extrair contexto de sizing
# ---------------------------------------------------------------------------

def _extract_sizing_context(sizing_result) -> dict:
    """Extrai parâmetros relevantes de um resultado de sizing.

    Suporta objetos com atributos (dataclass) ou dicts.

    Parâmetros
    ----------
    sizing_result : object | dict

    Retorna
    -------
    dict com parâmetros extraídos (valores None para os ausentes)
    """
    keys = [
        "ns", "eta_total", "eta_hid", "npsh_r", "npsh_r_m", "npsh_a",
        "d2_mm", "b2_mm", "beta2", "beta2_deg", "u2", "n_rpm", "q_m3h",
        "h_m", "p_kw", "p_shaft_kw",
    ]

    context: dict = {}

    if isinstance(sizing_result, dict):
        for k in keys:
            if k in sizing_result:
                context[k] = sizing_result[k]
    else:
        for k in keys:
            val = getattr(sizing_result, k, None)
            if val is not None:
                context[k] = val

    return context
