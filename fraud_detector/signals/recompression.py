"""Error Level Analysis (ELA) global: diferença entre a imagem e sua recompressão JPEG."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding


def ela_image(image: Image.Image, quality: int = 90) -> tuple[Image.Image, float, float]:
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    difference = ImageChops.difference(rgb, recompressed)
    values = np.asarray(difference, dtype=np.float32)
    mean_error = float(values.mean())
    p95_error = float(np.percentile(values, 95))
    extrema = difference.getextrema()
    maximum = max(channel[1] for channel in extrema) or 1
    rendered = ImageEnhance.Brightness(difference).enhance(255.0 / maximum)
    return rendered, mean_error, p95_error


class RecompressionSignal:
    name = "recompression"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        rendered, ela_mean, ela_p95 = ela_image(ctx.rgb, config.ela_quality)
        result.images["ela"] = rendered
        result.features["ela_mean"] = round(ela_mean, 3)
        result.features["ela_p95"] = round(ela_p95, 3)
        result.details["ela"] = {"mean": round(ela_mean, 3), "p95": round(ela_p95, 3),
                                 "quality": config.ela_quality}
        if ela_mean >= config.ela_high_mean:
            result.add(finding("ela_high", "Inconsistência forte de recompressão", 30,
                               f"ELA médio: {ela_mean:.2f}; percentil 95: {ela_p95:.2f}.", "alto"))
        elif ela_mean >= config.ela_warn_mean:
            result.add(finding("ela_warn", "Inconsistência de recompressão", 15,
                               f"ELA médio: {ela_mean:.2f}; percentil 95: {ela_p95:.2f}.", "médio"))
        return result
