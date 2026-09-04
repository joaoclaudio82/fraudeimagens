"""Configuração única e serializável da análise.

Todos os limiares vivem aqui para que a interface, a API, o CLI e o harness de
avaliação compartilhem exatamente os mesmos parâmetros e o relatório de auditoria
registre com que configuração cada decisão foi tomada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    # Decisão / encaminhamento
    review_threshold: int = 35
    attention_threshold: int = 15

    # Qualidade
    blur_warn_variance: float = 80.0
    min_width: int = 700
    min_height: int = 500

    # ELA global (mantido como indicador legado e para o mapa visual)
    ela_quality: int = 90
    ela_warn_mean: float = 7.0
    ela_high_mean: float = 14.0

    # Duplicidade
    duplicate_hamming_distance: int = 7

    # OCR
    ocr_languages: str = "por+eng"
    ocr_psm: int = 6

    def with_overrides(self, **overrides: Any) -> "AnalysisConfig":
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "AnalysisConfig":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in (values or {}).items() if key in known})
