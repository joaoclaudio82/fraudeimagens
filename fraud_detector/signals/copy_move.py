"""Copy-move: uma região da própria imagem copiada e colada em outro lugar.

É a forma mais comum de alterar um dígito ou um bloco de texto em um comprovante:
copia-se um "5" de outra posição, ou um trecho de linha inteira. Keypoints ORB
casados dentro da mesma imagem revelam pares de pontos com o mesmo deslocamento;
a correlação entre os dois recortes confirma que a área inteira foi copiada, e
não apenas um glifo que se repete naturalmente (o "R$" de cada linha, por exemplo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding, region_box


@dataclass
class CopyMoveCluster:
    size: int
    shift: tuple[float, float]
    source: dict[str, int]
    target: dict[str, int]
    correlation: float
    agreement: float = 0.0
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    occurrences: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "matches": self.size,
            "shift": [round(self.shift[0], 1), round(self.shift[1], 1)],
            "source": self.source,
            "target": self.target,
            "correlation": round(self.correlation, 3),
            "cell_agreement": round(self.agreement, 3),
            "occurrences": self.occurrences,
        }


@dataclass
class CopyMoveAnalysis:
    keypoints: int = 0
    candidate_pairs: int = 0
    clusters: list[CopyMoveCluster] = field(default_factory=list)
    confirmed: list[CopyMoveCluster] = field(default_factory=list)
    note: str = ""


def _bbox(points: list[tuple[float, float]], width: int, height: int, margin: int = 6) -> dict[str, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(min(xs)) - margin)
    y0 = max(0, int(min(ys)) - margin)
    x1 = min(width, int(max(xs)) + margin + 1)
    y1 = min(height, int(max(ys)) + margin + 1)
    return region_box(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _shifted(box: dict[str, int], shift: tuple[float, float], width: int, height: int) -> dict[str, int]:
    x0 = int(round(box["x"] + shift[0]))
    y0 = int(round(box["y"] + shift[1]))
    x0 = min(max(0, x0), max(0, width - box["w"]))
    y0 = min(max(0, y0), max(0, height - box["h"]))
    return region_box(x0, y0, min(box["w"], width - x0), min(box["h"], height - y0))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 64 or a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def _patches(gray: np.ndarray, source: dict[str, int], target: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    a = gray[source["y"]:source["y"] + source["h"], source["x"]:source["x"] + source["w"]]
    b = gray[target["y"]:target["y"] + target["h"], target["x"]:target["x"] + target["w"]]
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w].astype(np.float32), b[:h, :w].astype(np.float32)


def _correlation(gray: np.ndarray, source: dict[str, int], target: dict[str, int]) -> float:
    a, b = _patches(gray, source, target)
    if a.shape[0] < 8 or a.shape[1] < 8:
        return 0.0
    return _ncc(a, b)


def _cell_agreement(gray: np.ndarray, source: dict[str, int], target: dict[str, int], cell: int,
                    min_ncc: float = 0.9) -> float:
    """Fração das células texturizadas do recorte que também casam individualmente.

    Uma cópia integral casa em todas as células; uma tabela com rótulos repetidos
    casa só nas células dos rótulos e falha nas dos valores, que mudam a cada linha.
    """
    a, b = _patches(gray, source, target)
    h, w = a.shape
    if h < cell or w < cell:
        return _ncc(a, b) >= min_ncc and 1.0 or 0.0
    total = agree = 0
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            block_a = a[y:y + cell, x:x + cell]
            if block_a.std() < 2.0:
                continue
            total += 1
            if _ncc(block_a, b[y:y + cell, x:x + cell]) >= min_ncc:
                agree += 1
    return agree / total if total else 0.0


def _occurrences(gray: np.ndarray, source: dict[str, int], threshold: float, limit: int = 6) -> int:
    """Quantas vezes o recorte aparece na imagem além de si mesmo.

    Cópia forjada: exatamente uma outra ocorrência. Rótulo de tabela, logotipo ou
    padrão de formulário: várias ocorrências, estrutura repetitiva legítima.
    """
    patch = gray[source["y"]:source["y"] + source["h"], source["x"]:source["x"] + source["w"]].astype(np.float32)
    if patch.shape[0] < 8 or patch.shape[1] < 8 or patch.std() < 1e-3:
        return 0
    response = cv2.matchTemplate(gray.astype(np.float32), patch, cv2.TM_CCOEFF_NORMED)
    h, w = patch.shape
    count = 0
    for _ in range(limit + 1):
        _, value, _, (x, y) = cv2.minMaxLoc(response)
        if value < threshold:
            break
        if abs(x - source["x"]) >= w / 2 or abs(y - source["y"]) >= h / 2:
            count += 1
        response[max(0, y - h // 2):y + h // 2 + 1, max(0, x - w // 2):x + w // 2 + 1] = -1.0
    return count


def detect_copy_move(gray: np.ndarray, config: AnalysisConfig) -> CopyMoveAnalysis:
    analysis = CopyMoveAnalysis()
    height, width = gray.shape
    orb = cv2.ORB_create(nfeatures=config.copy_move_features, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    analysis.keypoints = len(keypoints)
    if descriptors is None or len(keypoints) < 20:
        analysis.note = "poucos pontos de interesse"
        return analysis

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs: dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float]]] = {}
    for candidates in matcher.knnMatch(descriptors, descriptors, k=3):
        others = [m for m in candidates if m.trainIdx != m.queryIdx]
        if not others:
            continue
        best = others[0]
        if best.distance > config.copy_move_max_hamming:
            continue
        if len(others) > 1 and best.distance > config.copy_move_ratio * others[1].distance:
            continue
        p = keypoints[best.queryIdx].pt
        q = keypoints[best.trainIdx].pt
        if math.hypot(p[0] - q[0], p[1] - q[1]) < config.copy_move_min_shift:
            continue
        key = (min(best.queryIdx, best.trainIdx), max(best.queryIdx, best.trainIdx))
        pairs.setdefault(key, (p, q) if best.queryIdx < best.trainIdx else (q, p))
    analysis.candidate_pairs = len(pairs)
    if not pairs:
        return analysis

    # Agrupa pares pelo vetor de deslocamento (cópia rígida => mesmo deslocamento para todos os pontos).
    clusters: list[dict[str, Any]] = []
    tolerance = config.copy_move_shift_tolerance
    for p, q in pairs.values():
        shift = (q[0] - p[0], q[1] - p[1])
        for cluster in clusters:
            mean = cluster["mean"]
            if abs(shift[0] - mean[0]) <= tolerance and abs(shift[1] - mean[1]) <= tolerance:
                cluster["items"].append((p, q))
                n = len(cluster["items"])
                cluster["mean"] = ((mean[0] * (n - 1) + shift[0]) / n, (mean[1] * (n - 1) + shift[1]) / n)
                break
        else:
            clusters.append({"mean": shift, "items": [(p, q)]})

    max_area = config.copy_move_max_region_fraction * width * height
    min_area = config.copy_move_min_region_fraction * width * height
    seen_shifts: list[tuple[float, float]] = []
    for cluster in sorted(clusters, key=lambda c: -len(c["items"])):
        items = cluster["items"]
        if len(items) < config.copy_move_min_matches:
            break
        shift = cluster["mean"]
        # A mesma cópia aparece duas vezes: origem->cópia e cópia->origem. Basta uma.
        if any(abs(shift[0] + s[0]) <= tolerance and abs(shift[1] + s[1]) <= tolerance for s in seen_shifts):
            continue
        seen_shifts.append(shift)
        source = _bbox([p for p, _ in items], width, height)
        area = source["w"] * source["h"]
        if area > max_area or area < min_area:
            continue  # estrutura repetitiva ampla, ou glifo isolado que se repete naturalmente
        target = _shifted(source, cluster["mean"], width, height)
        correlation = _correlation(gray, source, target)
        agreement = _cell_agreement(gray, source, target, config.copy_move_cell)
        entry = CopyMoveCluster(len(items), cluster["mean"], source, target, correlation, agreement, items)
        if correlation >= config.copy_move_min_correlation and agreement >= config.copy_move_min_cell_agreement:
            entry.occurrences = _occurrences(gray, source, config.copy_move_min_correlation)
            if entry.occurrences == 1:
                analysis.confirmed.append(entry)
        analysis.clusters.append(entry)
    return analysis


def render_copy_move_overlay(rgb: Image.Image, analysis: CopyMoveAnalysis) -> Image.Image:
    canvas = np.asarray(rgb).copy()
    for cluster in analysis.confirmed[:3]:
        for box, color in ((cluster.source, (0, 200, 255)), (cluster.target, (255, 60, 60))):
            cv2.rectangle(canvas, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), color, 3)
        for p, q in cluster.pairs[:60]:
            cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), (255, 255, 0), 1)
    return Image.fromarray(canvas)


class CopyMoveSignal:
    name = "copy_move"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        if not config.copy_move_enabled:
            result.features["copy_move_found"] = False
            return result

        analysis = detect_copy_move(ctx.gray, config)
        best = analysis.confirmed[0] if analysis.confirmed else None
        result.features["keypoints"] = analysis.keypoints
        result.features["candidate_pairs"] = analysis.candidate_pairs
        result.features["best_cluster"] = max((c.size for c in analysis.clusters), default=0)
        result.features["best_correlation"] = round(max((c.correlation for c in analysis.clusters), default=0.0), 3)
        result.features["copy_move_found"] = best is not None
        result.details["clusters"] = [c.summary() for c in analysis.clusters[:5]]
        result.details["note"] = analysis.note
        result.images["overlay"] = render_copy_move_overlay(ctx.rgb, analysis)

        if best is not None:
            result.add(finding(
                "copy_move", "Região duplicada dentro da própria imagem", 30,
                f"{best.size} pontos casados com o mesmo deslocamento ({best.shift[0]:.0f}, {best.shift[1]:.0f}) px; "
                f"correlação {best.correlation:.2f} e {best.agreement:.0%} das células coincidem. Origem em x={best.source['x']}, "
                f"y={best.source['y']}; cópia em x={best.target['x']}, y={best.target['y']} "
                f"({best.target['w']}×{best.target['h']} px).",
                "alto", region=best.target))
        return result
