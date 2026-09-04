"""Combinação dos indicadores em um score de prioridade de revisão."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import AnalysisConfig


@dataclass
class ScoreResult:
    score: int
    decision: str
    model: str
    explanation: list[dict[str, Any]] = field(default_factory=list)


def decide(score: int, config: AnalysisConfig) -> str:
    if score >= config.review_threshold:
        return "REVISAR"
    if score >= config.attention_threshold:
        return "ATENÇÃO"
    return "BAIXO RISCO"


class AdditiveScorer:
    """Soma dos pontos de cada indicador, limitada a 100. Simples e totalmente explicável."""

    name = "additive"

    def score(self, findings: list[dict[str, Any]], features: dict[str, Any], config: AnalysisConfig) -> ScoreResult:
        raw = sum(int(item.get("points", 0)) for item in findings)
        score = max(0, min(100, raw))
        explanation = [{"source": item["code"], "contribution": item["points"], "label": item["label"]}
                       for item in sorted(findings, key=lambda item: -item.get("points", 0))]
        return ScoreResult(score=score, decision=decide(score, config), model=self.name, explanation=explanation)
