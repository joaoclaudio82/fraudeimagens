"""Consistência tipográfica por linha, a partir das caixas de palavra do OCR.

Um dígito ou palavra reescrito em editor raramente reproduz a altura e a linha
de base do texto original impresso. Dentro de uma mesma linha, palavras em caixa
alta ou numéricas têm altura de caixa muito parecida; uma palavra bem maior, menor
ou deslocada verticalmente merece inspeção.

Limitações: depende da qualidade das caixas do Tesseract; palavras minúsculas com
descendentes (g, j, p, q, y) têm altura e base diferentes por natureza e ficam
fora da comparação. É um sinal fraco, pensado para somar com os demais.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding, region_box

ASCENDERS = set("bdfhklt")
DESCENDERS = set("gjpqy")


def comparable(text: str) -> bool:
    """Palavras cuja caixa vai da linha de base até a altura de caixa alta (ou ascendente)."""
    alnum = [ch for ch in text if ch.isalnum()]
    if len(alnum) < 2:
        return False
    lowers = [ch for ch in text if ch.isalpha() and ch.islower()]
    if not lowers:
        return True
    if any(ch in DESCENDERS for ch in lowers):
        return False
    return any(ch in ASCENDERS for ch in lowers) or any(ch.isupper() or ch.isdigit() for ch in text)


def line_outliers(words: list[dict[str, Any]], config: AnalysisConfig) -> tuple[bool, list[dict[str, Any]]]:
    usable = [w for w in words if w.get("conf", 0) >= config.typography_min_conf and comparable(w["text"])]
    if len(usable) < 3:
        return False, []
    heights = np.array([w["height"] for w in usable], dtype=float)
    bottoms = np.array([w["top"] + w["height"] for w in usable], dtype=float)
    median_height = float(np.median(heights))
    median_bottom = float(np.median(bottoms))
    if median_height <= 0:
        return False, []

    outliers: list[dict[str, Any]] = []
    for word, height, bottom in zip(usable, heights, bottoms):
        height_dev = (height - median_height) / median_height
        baseline_dev = (bottom - median_bottom) / median_height
        if abs(height_dev) >= config.typography_height_tolerance or abs(baseline_dev) >= config.typography_baseline_tolerance:
            outliers.append({
                "text": word["text"],
                "height_dev": round(float(height_dev), 2),
                "baseline_dev": round(float(baseline_dev), 2),
                "region": region_box(word["left"], word["top"], word["width"], word["height"]),
            })
    # Se a maioria destoa, a linha é heterogênea por natureza (título + corpo, colunas); não conclui nada.
    if len(outliers) > len(usable) / 2:
        return True, []
    return True, outliers


class TypographySignal:
    name = "typography"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        words = ctx.shared.get("ocr_words") or []
        result.features["typography_checked"] = False
        result.features["typography_outliers"] = 0
        result.features["typography_max_dev"] = 0.0
        if len(words) < config.typography_min_words:
            result.details["note"] = "palavras insuficientes"
            return result

        lines: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for word in words:
            lines.setdefault(tuple(word.get("line", (0, 0, 0))), []).append(word)

        checked = 0
        outliers: list[dict[str, Any]] = []
        for line_words in lines.values():
            was_checked, found = line_outliers(line_words, config)
            checked += int(was_checked)
            outliers.extend(found)

        result.features["typography_checked"] = checked > 0
        result.features["typography_lines"] = checked
        result.features["typography_outliers"] = len(outliers)
        result.features["typography_max_dev"] = round(
            max((max(abs(o["height_dev"]), abs(o["baseline_dev"])) for o in outliers), default=0.0), 2)
        result.details["outliers"] = outliers
        if not outliers:
            return result

        worst = max(outliers, key=lambda o: max(abs(o["height_dev"]), abs(o["baseline_dev"])))
        described = ", ".join(
            f"'{o['text']}' (altura {o['height_dev']:+.0%}, base {o['baseline_dev']:+.0%})" for o in outliers[:3])
        result.add(finding(
            "typography_inconsistent", "Palavra com tipografia diferente da própria linha", 10,
            f"{len(outliers)} palavra(s) com altura ou linha de base fora do padrão da linha: {described}.",
            "médio", region=worst["region"]))
        return result
