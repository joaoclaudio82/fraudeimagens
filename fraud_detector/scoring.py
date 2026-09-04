"""Combinação dos indicadores em um score de prioridade de revisão.

Dois modelos:
- AdditiveScorer: soma dos pontos de cada indicador (limitada a 100). Simples,
  explicável, sem dados. É o padrão.
- LogisticScorer: regressão logística sobre as features dos sinais, treinada
  com fraud_detector.evaluation.train a partir de uma base rotulada. Continua
  explicável: a contribuição de cada feature é coeficiente × valor padronizado.
  Um modelo treinado em base sintética precisa ser reavaliado em dados reais
  antes de decidir sozinho; por isso existe o modo "max", que usa o maior dos
  dois scores e nunca é menos conservador que o aditivo.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AnalysisConfig

EXCLUDED_FEATURES = {"duplicates.sha256", "duplicates.perceptual_hash", "duplicates.phash"}


@dataclass
class ScoreResult:
    score: int
    decision: str
    model: str
    explanation: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


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


def feature_value(features: dict[str, Any], name: str) -> float | None:
    """Valor numérico de uma feature do relatório; categóricas usam a forma 'nome=valor'."""
    if "=" in name:
        base, _, expected = name.partition("=")
        if base not in features:
            return None
        return 1.0 if str(features[base]) == expected else 0.0
    value = features.get(name)
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LogisticScorer:
    """Regressão logística sobre as features, com explicação por contribuição."""

    def __init__(self, model: dict[str, Any], label: str = "logistic") -> None:
        if model.get("model") != "logistic":
            raise ValueError("modelo inválido: esperado 'logistic'")
        self.model = model
        self.name = label
        self.features: list[str] = list(model["features"])
        self.mean = [float(v) for v in model["mean"]]
        self.std = [float(v) or 1.0 for v in model["std"]]
        self.coef = [float(v) for v in model["coef"]]
        self.intercept = float(model["intercept"])

    @classmethod
    def from_file(cls, path: str | Path) -> "LogisticScorer":
        path = Path(path)
        model = json.loads(path.read_text(encoding="utf-8"))
        return cls(model, label=f"logistic:{path.name}")

    def probability(self, features: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
        logit = self.intercept
        contributions: list[dict[str, Any]] = []
        for index, name in enumerate(self.features):
            raw = feature_value(features, name)
            value = self.mean[index] if raw is None else raw
            z = (value - self.mean[index]) / self.std[index]
            contribution = self.coef[index] * z
            logit += contribution
            if raw is not None and abs(contribution) > 1e-6:
                contributions.append({"source": name, "value": raw, "contribution": round(contribution, 4)})
        contributions.sort(key=lambda item: -abs(item["contribution"]))
        return 1.0 / (1.0 + math.exp(-logit)), contributions

    def score(self, findings: list[dict[str, Any]], features: dict[str, Any], config: AnalysisConfig) -> ScoreResult:
        probability, contributions = self.probability(features)
        score = int(round(100 * probability))
        return ScoreResult(score=score, decision=decide(score, config), model=self.name,
                           explanation=contributions[:8], extra={"probability": round(probability, 4)})


class MaxScorer:
    """Usa o maior entre o score aditivo e o do modelo: nunca menos conservador que o aditivo."""

    name = "max"

    def __init__(self, learned: LogisticScorer) -> None:
        self.learned = learned
        self.additive = AdditiveScorer()

    def score(self, findings: list[dict[str, Any]], features: dict[str, Any], config: AnalysisConfig) -> ScoreResult:
        a = self.additive.score(findings, features, config)
        b = self.learned.score(findings, features, config)
        chosen, other = (a, b) if a.score >= b.score else (b, a)
        return ScoreResult(score=chosen.score, decision=decide(chosen.score, config),
                           model=f"max({a.model}, {b.model})", explanation=chosen.explanation,
                           extra={"additive_score": a.score, "learned_score": b.score, **b.extra})


def build_scorer(config: AnalysisConfig) -> Any:
    """Escolhe o scorer pela configuração; sem modelo (ou arquivo ausente) cai no aditivo."""
    path = config.scoring_model
    if not path or not Path(path).exists():
        return AdditiveScorer()
    learned = LogisticScorer.from_file(path)
    if config.scoring_mode == "max":
        return MaxScorer(learned)
    return learned
