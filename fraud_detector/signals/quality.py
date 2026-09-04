"""Nitidez e resolução: evidência insuficiente não é fraude, mas reduz a confiança."""

from __future__ import annotations

import cv2

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding


class QualitySignal:
    name = "quality"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        blur_variance = float(cv2.Laplacian(ctx.gray, cv2.CV_64F).var())
        result.features["blur_variance"] = round(blur_variance, 2)
        result.features["width"] = ctx.width
        result.features["height"] = ctx.height
        result.features["megapixels"] = round(ctx.width * ctx.height / 1e6, 3)

        if blur_variance < config.blur_warn_variance:
            result.add(finding("blur", "Imagem pouco nítida", 10,
                               f"Variância do Laplaciano: {blur_variance:.1f}.", "baixo"))
        if ctx.width < config.min_width or ctx.height < config.min_height:
            result.add(finding("resolution", "Resolução baixa", 8,
                               f"Imagem com {ctx.width} × {ctx.height} pixels.", "baixo"))
        return result
