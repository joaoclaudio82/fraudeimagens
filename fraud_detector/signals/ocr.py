"""OCR local com Tesseract e conferência dos campos operacionais esperados."""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from ..config import AnalysisConfig
from ..context import ImageContext
from .base import SignalResult, finding


def run_tesseract(image: Image.Image, config: AnalysisConfig) -> tuple[str, str | None]:
    try:
        import pytesseract

        text = pytesseract.image_to_string(image, lang=config.ocr_languages, config=f"--psm {config.ocr_psm}")
        return text.strip(), None
    except Exception as exc:
        return "", f"OCR indisponível: {type(exc).__name__}. Instale o Tesseract com os idiomas por/eng."


def text_checks(text: str, expected: dict[str, str] | None) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    if not expected:
        return [], {}
    normalized = re.sub(r"\s+", " ", text.casefold())
    findings: list[dict[str, Any]] = []
    matches: dict[str, bool] = {}
    for field, raw_value in expected.items():
        value = str(raw_value).strip()
        if not value:
            continue
        compact_value = re.sub(r"\s+", " ", value.casefold())
        matched = compact_value in normalized
        matches[field] = matched
        if not matched:
            findings.append(finding(
                f"field_{field}", f"Campo esperado não localizado: {field}", 12,
                f"Valor informado '{value}' não foi encontrado pelo OCR.", "médio"))
    return findings, matches


class OcrSignal:
    name = "ocr"

    def run(self, ctx: ImageContext, config: AnalysisConfig) -> SignalResult:
        result = SignalResult(self.name)
        text, warning = run_tesseract(ctx.rgb, config)
        result.details["text"] = text
        result.details["warning"] = warning
        result.features["ocr_available"] = warning is None
        result.features["ocr_chars"] = len(text)

        text_findings, matches = text_checks(text, ctx.expected)
        result.details["field_matches"] = matches
        result.features["fields_checked"] = len(matches)
        result.features["fields_missing"] = sum(1 for ok in matches.values() if not ok)
        for item in text_findings:
            result.add(item)
        ctx.shared["ocr_text"] = text
        return result
