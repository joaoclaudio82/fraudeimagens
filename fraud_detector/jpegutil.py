"""Utilidades sobre o contêiner JPEG: tabelas de quantização, qualidade estimada e subamostragem.

As tabelas de quantização são uma assinatura barata do último software que salvou
o arquivo. Câmeras, libjpeg/Pillow e a maioria dos apps usam as tabelas padrão
IJG escaladas por um fator de qualidade; alguns editores usam tabelas próprias.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image

STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)

STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
], dtype=np.float64)

SUBSAMPLING_NAMES = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def quantization_tables(image: Image.Image) -> dict[int, list[int]]:
    tables = getattr(image, "quantization", None) or {}
    return {int(key): [int(v) for v in value] for key, value in tables.items()}


def scaled_standard_table(quality: float, reference: np.ndarray = STD_LUMA) -> np.ndarray:
    quality = float(min(100, max(1, quality)))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((reference * scale + 50) / 100), 1, 255)


def estimate_quality(table: Sequence[int], reference: np.ndarray = STD_LUMA) -> float:
    values = np.asarray(table, dtype=np.float64)
    if values.size != reference.size:
        raise ValueError("tabela de quantização deve ter 64 coeficientes")
    ratio = float((values / reference).mean() * 100)
    quality = (200 - ratio) / 2 if ratio <= 100 else 5000 / ratio
    return float(min(100.0, max(1.0, quality)))


def is_standard_table(table: Sequence[int], quality: float, reference: np.ndarray = STD_LUMA,
                      tolerance: int = 1) -> bool:
    values = np.asarray(table, dtype=np.float64)
    expected = scaled_standard_table(round(quality), reference)
    return bool(np.abs(values - expected).max() <= tolerance)


def subsampling(image: Image.Image) -> str | None:
    try:
        from PIL import JpegImagePlugin

        code = JpegImagePlugin.get_sampling(image)
    except Exception:
        return None
    return SUBSAMPLING_NAMES.get(code)


def jpeg_summary(image: Image.Image) -> dict[str, Any]:
    """Resumo serializável do contêiner JPEG (vazio para outros formatos)."""
    if (image.format or "").upper() != "JPEG":
        return {"is_jpeg": False}
    tables = quantization_tables(image)
    luma = tables.get(0)
    chroma = tables.get(1)
    summary: dict[str, Any] = {
        "is_jpeg": True,
        "tables": len(tables),
        "estimated_quality": None,
        "standard_tables": None,
        "subsampling": subsampling(image),
        "progressive": bool(image.info.get("progressive") or image.info.get("progression")),
    }
    if luma and len(luma) == 64:
        quality = estimate_quality(luma)
        summary["estimated_quality"] = int(round(quality))
        standard = is_standard_table(luma, quality)
        if chroma and len(chroma) == 64:
            standard = standard and is_standard_table(chroma, quality, STD_CHROMA)
        summary["standard_tables"] = standard
    return summary
