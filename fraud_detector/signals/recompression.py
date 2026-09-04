"""Recompressão JPEG: ELA global (legado) e análise localizada por bloco (JPEG ghost).

A média global do ELA não muda quando só uma linha do comprovante é editada. A
análise por bloco procura o "fantasma" da primeira compressão: blocos que já
foram salvos na qualidade Q têm erro de recompressão menor exatamente em Q. Uma
região colada ou reescrita depois da primeira compressão não carrega esse
histórico e destoa do restante da imagem, o que permite apontar onde olhar.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from ..config import AnalysisConfig
from ..context import ImageContext
from ..jpegutil import jpeg_summary
from .base import SignalResult, finding, region_box


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


@dataclass
class GhostAnalysis:
    applicable: bool
    block: int
    found: bool = False
    ghost_quality: int | None = None
    last_quality: int | None = None
    median_dip: float = 0.0
    threshold: float = 0.0
    coverage: float = 0.0
    regions: list[dict[str, int]] = field(default_factory=list)
    dip_map: np.ndarray | None = None
    textured: np.ndarray | None = None
    suspicious: np.ndarray | None = None
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "found": self.found,
            "ghost_quality": self.ghost_quality,
            "last_quality": self.last_quality,
            "median_dip": round(float(self.median_dip), 4),
            "threshold": round(float(self.threshold), 4),
            "coverage": round(float(self.coverage), 4),
            "regions": self.regions,
            "block": self.block,
            "note": self.note,
        }


def _block_reduce(values: np.ndarray, block: int) -> np.ndarray:
    h, w = values.shape
    H, W = h // block * block, w // block * block
    return values[:H, :W].reshape(H // block, block, W // block, block).mean(axis=(1, 3))


def _block_std(gray: np.ndarray, block: int) -> np.ndarray:
    h, w = gray.shape
    H, W = h // block * block, w // block * block
    tiles = gray[:H, :W].astype(np.float32).reshape(H // block, block, W // block, block)
    return tiles.std(axis=(1, 3))


def recompression_errors(rgb: Image.Image, qualities: tuple[int, ...], block: int) -> np.ndarray:
    """Erro quadrático médio por bloco entre a imagem e sua recompressão em cada qualidade."""
    arr = np.asarray(rgb, dtype=np.float32)
    out = []
    for quality in qualities:
        buffer = io.BytesIO()
        rgb.save(buffer, "JPEG", quality=int(quality))
        buffer.seek(0)
        rec = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
        out.append(_block_reduce(((arr - rec) ** 2).mean(axis=2), block))
    return np.stack(out)


def jpeg_ghost(rgb: Image.Image, gray: np.ndarray, config: AnalysisConfig,
               last_quality: int | None = None) -> GhostAnalysis:
    block = config.ghost_block
    qualities = tuple(sorted(int(q) for q in config.ghost_qualities))
    if len(qualities) < 3 or min(rgb.size) < 4 * block:
        return GhostAnalysis(applicable=False, block=block, note="imagem pequena demais ou qualidades insuficientes")

    errors = recompression_errors(rgb, qualities, block)
    std = _block_std(gray, block)
    # Blocos planos não têm erro para medir; blocos de contraste altíssimo (bordas do papel,
    # texto, assinatura) têm erro dominado por ringing e não exibem o ghost mesmo íntegros.
    textured = (std >= config.ghost_min_texture) & (std <= config.ghost_max_texture)
    analysis = GhostAnalysis(applicable=True, block=block, last_quality=last_quality, textured=textured)
    if textured.sum() < config.ghost_min_blocks * 4:
        analysis.note = "textura insuficiente para estimar o histórico de compressão"
        return analysis

    eps = 0.05
    candidates: list[tuple[int, float, np.ndarray]] = []
    for index in range(1, len(qualities) - 1):
        quality = qualities[index]
        current = errors[index]
        # Recomprimir na qualidade do último salvamento devolve erro ~0 em todo bloco:
        # não é um ghost, é a própria compressão atual. Fica de fora da busca.
        degenerate = float(np.median(current[textured])) < 0.02
        near_last = last_quality is not None and abs(quality - last_quality) < 3
        if degenerate or near_last:
            continue
        neighbours = (errors[index - 1] + errors[index + 1]) / 2.0
        dip = (neighbours - current) / (current + eps)
        candidates.append((quality, float(np.median(dip[textured])), dip))

    if not candidates:
        analysis.note = "sem qualidades candidatas fora do último salvamento"
        return analysis

    quality, median_dip, dip = max(candidates, key=lambda item: item[1])
    analysis.ghost_quality = quality
    analysis.median_dip = median_dip
    analysis.dip_map = dip
    if median_dip < config.ghost_min_dip:
        analysis.note = "nenhum histórico de compressão anterior detectável"
        return analysis

    # Limiar robusto: abaixo de uma fração da mediana E fora da dispersão normal (MAD) dos blocos.
    # Um ghost fraco e ruidoso tem dispersão grande e não produz suspeitos; uma edição real tem
    # queda ~0 em um ghost forte e bem definido.
    mad = float(np.median(np.abs(dip[textured] - median_dip)))
    threshold = min(config.ghost_ratio * median_dip, median_dip - config.ghost_mad_k * 1.4826 * mad)
    analysis.threshold = threshold
    suspicious = textured & (dip < threshold)
    analysis.suspicious = suspicious
    analysis.coverage = float(suspicious.sum() / max(1, textured.sum()))

    count, _, stats, _ = cv2.connectedComponentsWithStats(suspicious.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        fill = area / float(w * h)
        if area >= config.ghost_min_blocks and fill >= config.ghost_min_fill:
            regions.append((area, region_box(x * block, y * block, w * block, h * block)))
    regions.sort(key=lambda item: -item[0])
    analysis.regions = [box for _, box in regions[:5]]
    analysis.found = bool(analysis.regions)
    return analysis


def render_ghost_overlay(rgb: Image.Image, analysis: GhostAnalysis) -> Image.Image:
    base = np.asarray(rgb).copy()
    if analysis.dip_map is None or analysis.textured is None:
        return Image.fromarray(base)
    block = analysis.block
    strength = np.clip(1.0 - analysis.dip_map / (analysis.median_dip + 1e-6), 0.0, 1.0)
    strength[~analysis.textured] = 0.0
    heat = (strength * 255).astype(np.uint8)
    H, W = heat.shape[0] * block, heat.shape[1] * block
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
    base[:H, :W] = (0.55 * base[:H, :W] + 0.45 * color).astype(np.uint8)
    for box in analysis.regions:
        cv2.rectangle(base, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (255, 255, 255), 3)
    return Image.fromarray(base)


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

        if not config.ghost_enabled:
            result.features["ghost_found"] = False
            return result

        container = jpeg_summary(ctx.image)
        last_quality = container.get("estimated_quality") if container.get("is_jpeg") else None
        analysis = jpeg_ghost(ctx.rgb, ctx.gray, config, last_quality)
        result.details["ghost"] = analysis.summary()
        result.features["ghost_found"] = analysis.found
        result.features["ghost_median_dip"] = round(float(analysis.median_dip), 4)
        result.features["ghost_coverage"] = round(float(analysis.coverage), 4)
        result.features["ghost_regions"] = len(analysis.regions)
        result.features["ghost_largest_area"] = (
            analysis.regions[0]["w"] * analysis.regions[0]["h"] if analysis.regions else 0)
        result.images["ghost_overlay"] = render_ghost_overlay(ctx.rgb, analysis)

        if analysis.found and analysis.coverage <= config.ghost_max_coverage:
            largest = analysis.regions[0]
            result.add(finding(
                "ghost_local", "Região sem o histórico de compressão do restante da imagem", 30,
                f"Recompressão anterior estimada em qualidade {analysis.ghost_quality}; "
                f"{len(analysis.regions)} região(ões) destoam, cobrindo {analysis.coverage:.0%} da área texturizada. "
                f"Maior região em x={largest['x']}, y={largest['y']}, {largest['w']}×{largest['h']} px.",
                "alto", region=largest))
        elif analysis.found:
            result.add(finding(
                "ghost_global", "Histórico de compressão heterogêneo em grande parte da imagem", 8,
                f"{analysis.coverage:.0%} da área texturizada destoa; pode indicar recorte, redimensionamento "
                "ou montagem ampla, mas também múltiplos salvamentos.", "baixo"))
        return result
