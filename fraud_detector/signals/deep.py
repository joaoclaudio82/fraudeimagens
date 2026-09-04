"""Plugin para detectores de manipulação baseados em redes neurais (TruFor, CAT-Net, MVSS-Net...).

Esses modelos produzem um mapa de probabilidade de manipulação por pixel e uma
pontuação global. Os pesos e o código de cada um têm licenças e dependências
próprias (PyTorch, CUDA opcional), por isso não vêm embutidos: este módulo define
o contrato e um carregador por variável de ambiente, e o sinal fica desligado até
que um adaptador seja configurado.

Contrato do adaptador: um callable ``predict(rgb: PIL.Image) -> DeepPrediction``
(ou dict com as mesmas chaves). Configure com
``AnalysisConfig(deep_model="meu_pacote.modulo:funcao")`` ou
``IMAGEGUARD_DEEP_MODEL=meu_pacote.modulo:funcao``.

Antes de confiar em um detector profundo na triagem, meça-o no harness
(scripts/evaluate.py): modelos treinados em fotos naturais podem generalizar mal
para fotos de comprovantes, capturas e reenvios por WhatsApp.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding, region_box


@dataclass
class DeepPrediction:
    score: float                                   # 0..1, probabilidade global de manipulação
    mask: np.ndarray | None = None                 # HxW em 0..1, opcional
    model: str = "deep"
    details: dict[str, Any] = field(default_factory=dict)


Predictor = Callable[[Image.Image], DeepPrediction | dict[str, Any]]


def load_predictor(spec: str | None) -> Predictor | None:
    """Resolve 'pacote.modulo:atributo' para um callable, ou None se não configurado."""
    spec = spec or os.environ.get("IMAGEGUARD_DEEP_MODEL")
    if not spec:
        return None
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    predictor = getattr(module, attribute or "predict")
    if not callable(predictor):
        raise TypeError(f"{spec} não é chamável")
    return predictor


def _coerce(prediction: DeepPrediction | dict[str, Any]) -> DeepPrediction:
    if isinstance(prediction, DeepPrediction):
        return prediction
    return DeepPrediction(score=float(prediction["score"]), mask=prediction.get("mask"),
                          model=str(prediction.get("model", "deep")), details=dict(prediction.get("details", {})))


def mask_regions(mask: np.ndarray, size: tuple[int, int], threshold: float, min_area_fraction: float = 0.001
                 ) -> list[dict[str, int]]:
    """Caixas das regiões conectadas acima do limiar, na escala da imagem original."""
    width, height = size
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    binary = (resized >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area >= min_area_fraction * width * height:
            boxes.append((area, region_box(x, y, w, h)))
    boxes.sort(key=lambda item: -item[0])
    return [box for _, box in boxes[:5]]


def render_mask_overlay(rgb: Image.Image, mask: np.ndarray, regions: list[dict[str, int]]) -> Image.Image:
    base = np.asarray(rgb).copy()
    heat = cv2.resize((np.clip(mask, 0, 1) * 255).astype(np.uint8), rgb.size, interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
    blended = (0.55 * base + 0.45 * color).astype(np.uint8)
    for box in regions:
        cv2.rectangle(blended, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), (255, 255, 255), 3)
    return Image.fromarray(blended)


class DeepDetectorSignal:
    name = "deep"

    def __init__(self, predictor: Predictor | None = None) -> None:
        self._predictor = predictor
        self._resolved = predictor is not None

    def predictor(self, config: AnalysisConfig) -> Predictor | None:
        if not self._resolved:
            self._predictor = load_predictor(config.deep_model)
            self._resolved = True
        return self._predictor

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        predictor = self.predictor(config)
        result.features["deep_enabled"] = predictor is not None
        if predictor is None:
            result.details["note"] = "nenhum detector profundo configurado (deep_model / IMAGEGUARD_DEEP_MODEL)"
            return result

        prediction = _coerce(predictor(ctx.rgb))
        score = float(min(1.0, max(0.0, prediction.score)))
        result.features["deep_score"] = round(score, 4)
        result.details["model"] = prediction.model
        result.details.update(prediction.details)

        regions: list[dict[str, int]] = []
        if prediction.mask is not None:
            regions = mask_regions(prediction.mask, ctx.rgb.size, config.deep_mask_threshold)
            result.features["deep_regions"] = len(regions)
            result.features["deep_mask_mean"] = round(float(np.mean(prediction.mask)), 4)
            result.images["overlay"] = render_mask_overlay(ctx.rgb, prediction.mask, regions)
        result.details["regions"] = regions

        if score >= config.deep_high_threshold:
            result.add(finding("deep_manipulation", f"Detector {prediction.model}: manipulação provável", 30,
                               f"Probabilidade {score:.2f}" + (f"; {len(regions)} região(ões) destacada(s)." if regions else "."),
                               "alto", region=regions[0] if regions else None))
        elif score >= config.deep_warn_threshold:
            result.add(finding("deep_suspect", f"Detector {prediction.model}: indícios de manipulação", 12,
                               f"Probabilidade {score:.2f}.", "médio", region=regions[0] if regions else None))
        return result
