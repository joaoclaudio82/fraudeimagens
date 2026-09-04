"""OCR local com Tesseract, pré-processamento e conferência semântica dos campos esperados.

Além do texto, o sinal devolve as palavras com caixas e confiança (image_to_data),
que ficam em ``ctx.shared["ocr_words"]`` para a checagem tipográfica. A conferência
de campos usa fraud_detector.textnorm para comparar datas, valores, números e
códigos pelo significado, tolerando os erros típicos de OCR.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext
from ..textnorm import match_field
from .base import SignalResult, finding, region_box

OCR_UNAVAILABLE = "OCR indisponível: {error}. Instale o Tesseract com os idiomas por/eng."


def _rotate(image: np.ndarray, angle: float, interpolation: int, border_mode: int, border_value: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=interpolation, borderMode=border_mode, borderValue=border_value)


def estimate_skew(text_mask: np.ndarray, max_degrees: float, step: float = 0.5) -> float:
    """Ângulo de correção (graus) que deixa as linhas de texto horizontais.

    Método do perfil de projeção: gira a máscara de texto em ângulos candidatos e
    escolhe o que concentra mais pixels em poucas linhas (maior variância do perfil
    horizontal). Fundo uniforme e ruído não têm estrutura de linha e não enviesam.
    """
    if text_mask.size == 0 or not text_mask.any():
        return 0.0
    scale = min(1.0, 800.0 / text_mask.shape[1])
    small = cv2.resize(text_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else text_mask
    small = (small > 0).astype(np.uint8)

    candidates = sorted(np.arange(-max_degrees, max_degrees + step / 2, step), key=abs)
    best_angle, best_score = 0.0, -1.0
    for angle in candidates:
        rotated = _rotate(small, float(angle), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
        profile = rotated.sum(axis=1).astype(np.float64)
        score = float(profile.var())
        if score > best_score * 1.001:  # empate favorece o menor ângulo (lista ordenada por |ângulo|)
            best_angle, best_score = float(angle), score
    return best_angle if abs(best_angle) >= 0.5 else 0.0


def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def preprocess_for_ocr(np_rgb: np.ndarray, config: AnalysisConfig) -> tuple[Image.Image, dict[str, Any]]:
    """Cinza, ampliação, binarização adaptativa e correção de inclinação."""
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    info: dict[str, Any] = {"scale": 1.0, "angle": 0.0}
    if not config.ocr_preprocess:
        return Image.fromarray(gray), info

    if gray.shape[1] < config.ocr_upscale_min_width:
        info["scale"] = 2.0
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)
    binary = _binarize(gray)
    angle = estimate_skew(255 - binary, config.ocr_max_deskew_degrees)
    if abs(angle) >= 0.5:
        info["angle"] = round(angle, 2)
        gray = _rotate(gray, angle, cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
        binary = _binarize(gray)
    return Image.fromarray(binary), info


def _words_from_data(data: dict[str, list[Any]], scale: float) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    count = len(data.get("text", []))
    for index in range(count):
        text = str(data["text"][index]).strip()
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < 0:
            continue
        words.append({
            "text": text,
            "conf": conf,
            "left": int(round(int(data["left"][index]) / scale)),
            "top": int(round(int(data["top"][index]) / scale)),
            "width": int(round(int(data["width"][index]) / scale)),
            "height": int(round(int(data["height"][index]) / scale)),
            "line": (int(data.get("block_num", [0] * count)[index]),
                     int(data.get("par_num", [0] * count)[index]),
                     int(data.get("line_num", [0] * count)[index])),
        })
    return words


def text_from_words(words: list[dict[str, Any]]) -> str:
    lines: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for word in words:
        lines.setdefault(tuple(word["line"]), []).append(word)
    ordered = sorted(lines.values(), key=lambda items: (min(w["top"] for w in items), min(w["left"] for w in items)))
    return "\n".join(" ".join(w["text"] for w in sorted(items, key=lambda w: w["left"])) for items in ordered)


def run_tesseract(image: Image.Image, config: AnalysisConfig, scale: float = 1.0
                  ) -> tuple[str, list[dict[str, Any]], str | None]:
    """Devolve (texto, palavras com caixas, aviso). Nunca levanta exceção."""
    try:
        import pytesseract
    except Exception as exc:
        return "", [], OCR_UNAVAILABLE.format(error=type(exc).__name__)

    options = f"--psm {config.ocr_psm}"
    try:
        data = pytesseract.image_to_data(image, lang=config.ocr_languages, config=options,
                                         output_type=pytesseract.Output.DICT)
        words = _words_from_data(data, scale)
        return text_from_words(words), words, None
    except Exception as data_error:
        try:
            text = pytesseract.image_to_string(image, lang=config.ocr_languages, config=options)
            return text.strip(), [], None
        except Exception:
            return "", [], OCR_UNAVAILABLE.format(error=type(data_error).__name__)


def check_fields(text: str, expected: dict[str, str], config: AnalysisConfig
                 ) -> tuple[list[dict[str, Any]], dict[str, bool], dict[str, dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    matches: dict[str, bool] = {}
    details: dict[str, dict[str, Any]] = {}
    for field, raw_value in expected.items():
        value = str(raw_value).strip()
        if not value:
            continue
        result = match_field(field, value, text, config.ocr_fuzzy_threshold, config.ocr_code_threshold)
        matches[field] = result.matched
        details[field] = result.to_dict()
        if result.matched:
            continue
        if result.conflict:
            findings.append(finding(
                f"field_{field}", f"Campo divergente: {field}", 18,
                f"Esperado '{value}'; o OCR encontrou '{result.found}' ({result.kind}).", "alto"))
        else:
            findings.append(finding(
                f"field_{field}", f"Campo esperado não localizado: {field}", 8,
                f"Valor informado '{value}' não foi encontrado pelo OCR "
                f"(melhor similaridade {result.score:.0f}%).", "baixo"))
    return findings, matches, details


class OcrSignal:
    name = "ocr"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        prepared, info = preprocess_for_ocr(ctx.np_rgb, config)
        text, words, warning = run_tesseract(prepared, config, info["scale"])
        if not words and not text and warning is None and config.ocr_preprocess:
            # Binarização pode apagar texto em fotos muito escuras: tenta a imagem original.
            text, words, warning = run_tesseract(ctx.rgb, config, 1.0)
            info["fallback"] = "imagem original"

        result.details["text"] = text
        result.details["warning"] = warning
        result.details["preprocessing"] = info
        result.details["words"] = words
        result.features["ocr_available"] = warning is None
        result.features["ocr_chars"] = len(text)
        result.features["ocr_words"] = len(words)
        result.features["ocr_mean_conf"] = round(float(np.mean([w["conf"] for w in words])), 1) if words else 0.0
        result.features["ocr_skew"] = info["angle"]
        ctx.shared["ocr_text"] = text
        ctx.shared["ocr_words"] = words

        if warning is not None:
            result.details["field_matches"] = {}
            result.details["field_details"] = {}
            result.add(finding("ocr_unavailable", "OCR indisponível: campos não conferidos", 0, warning, "baixo"))
            return result

        if len(text) < config.ocr_min_text_chars:
            result.details["field_matches"] = {}
            result.details["field_details"] = {}
            result.add(finding("ocr_empty", "OCR não reconheceu texto legível", 5,
                               "Sem texto suficiente para conferir os campos; verifique foco, luz e enquadramento.",
                               "baixo"))
            return result

        text_findings, matches, details = check_fields(text, ctx.expected, config)
        result.details["field_matches"] = matches
        result.details["field_details"] = details
        result.features["fields_checked"] = len(matches)
        result.features["fields_matched"] = sum(1 for ok in matches.values() if ok)
        result.features["fields_conflict"] = sum(1 for item in details.values() if item["conflict"])
        result.features["fields_missing"] = sum(1 for field, ok in matches.items()
                                                if not ok and not details[field]["conflict"])
        for item in text_findings:
            result.add(item)
        return result
