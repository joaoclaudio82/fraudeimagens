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

    # Recompressão localizada (JPEG ghost por bloco)
    ghost_enabled: bool = True
    ghost_block: int = 16
    ghost_qualities: tuple[int, ...] = (50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
    ghost_min_texture: float = 1.5      # desvio padrão mínimo do bloco para participar
    ghost_min_dip: float = 0.3          # queda relativa mínima para aceitar que existe um ghost
    ghost_ratio: float = 0.25           # bloco suspeito: queda menor que esta fração da mediana
    ghost_min_blocks: int = 8           # tamanho mínimo de uma região conectada suspeita
    ghost_max_coverage: float = 0.5     # acima disso o sinal vira apenas informativo

    # Duplicidade
    duplicate_hamming_distance: int = 7

    # OCR
    ocr_languages: str = "por+eng"
    ocr_psm: int = 6
    ocr_preprocess: bool = True
    ocr_upscale_min_width: int = 1500   # abaixo disso a imagem é ampliada 2x antes do OCR
    ocr_max_deskew_degrees: float = 15.0
    ocr_fuzzy_threshold: int = 85       # similaridade mínima para texto livre (nomes, endereços)
    ocr_code_threshold: int = 90        # similaridade mínima para códigos e números longos
    ocr_min_text_chars: int = 5

    def with_overrides(self, **overrides: Any) -> "AnalysisConfig":
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "AnalysisConfig":
        known = {item.name for item in fields(cls)}
        cleaned: dict[str, Any] = {}
        for key, value in (values or {}).items():
            if key not in known:
                continue
            cleaned[key] = tuple(value) if isinstance(value, list) else value
        return cls(**cleaned)
