"""Contrato dos sinais plugáveis.

Cada sinal recebe o contexto da imagem e devolve, de forma isolada:
- ``features``: valores numéricos/categóricos usados por calibração e modelos;
- ``findings``: indicadores explicáveis com pontos, evidência textual e região opcional;
- ``details``: informações serializáveis para o relatório de auditoria;
- ``images``: artefatos visuais (mapas de calor) que não vão para o JSON.

Um sinal que falha não derruba a análise: o orquestrador registra o erro e segue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext

Severity = str  # "baixo" | "médio" | "alto"


def finding(
    code: str,
    label: str,
    points: int,
    evidence: str,
    severity: Severity,
    region: dict[str, int] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "code": code,
        "label": label,
        "points": points,
        "evidence": evidence,
        "severity": severity,
    }
    if region:
        item["region"] = region
    return item


def region_box(x: int, y: int, w: int, h: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


@dataclass
class SignalResult:
    name: str
    features: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    images: dict[str, Image.Image] = field(default_factory=dict)
    error: str | None = None

    def add(self, item: dict[str, Any]) -> None:
        self.findings.append(item)


class Signal(Protocol):
    name: str

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult: ...
