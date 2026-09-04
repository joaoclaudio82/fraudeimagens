from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageChops, ImageEnhance


@dataclass(frozen=True)
class AnalysisConfig:
    ela_quality: int = 90
    ela_warn_mean: float = 7.0
    ela_high_mean: float = 14.0
    blur_warn_variance: float = 80.0
    min_width: int = 700
    min_height: int = 500
    duplicate_hamming_distance: int = 7


def _risk(code: str, label: str, points: int, evidence: str, severity: str) -> dict[str, Any]:
    return {"code": code, "label": label, "points": points, "evidence": evidence, "severity": severity}


def _exif(image: Image.Image) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for key, value in image.getexif().items():
            name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = f"<{len(value)} bytes>"
            result[name] = str(value)
    except Exception:
        pass
    return result


def _ocr(image: Image.Image) -> tuple[str, str | None]:
    try:
        import pytesseract

        text = pytesseract.image_to_string(image, lang="por+eng", config="--psm 6")
        return text.strip(), None
    except Exception as exc:
        return "", f"OCR indisponível: {type(exc).__name__}. Instale o Tesseract com os idiomas por/eng."


def perceptual_hash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


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


def _text_checks(text: str, expected: dict[str, str] | None) -> tuple[list[dict[str, Any]], dict[str, bool]]:
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
            findings.append(_risk(
                f"field_{field}", f"Campo esperado não localizado: {field}", 12,
                f"Valor informado '{value}' não foi encontrado pelo OCR.", "médio"
            ))
    return findings, matches


def analyze_image(
    data: bytes,
    filename: str = "image",
    expected: dict[str, str] | None = None,
    known_hashes: list[str] | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    image = Image.open(io.BytesIO(data))
    image.load()
    rgb = image.convert("RGB")
    np_rgb = np.asarray(rgb)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    width, height = rgb.size
    exif = _exif(image)
    findings: list[dict[str, Any]] = []

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < config.blur_warn_variance:
        findings.append(_risk("blur", "Imagem pouco nítida", 10,
                              f"Variância do Laplaciano: {blur_variance:.1f}.", "baixo"))

    if width < config.min_width or height < config.min_height:
        findings.append(_risk("resolution", "Resolução baixa", 8,
                              f"Imagem com {width} × {height} pixels.", "baixo"))

    software = exif.get("Software", "")
    if software:
        findings.append(_risk("editing_software", "Software registrado nos metadados", 18,
                              f"EXIF Software: {software}", "médio"))

    ela_render, ela_mean, ela_p95 = ela_image(rgb, config.ela_quality)
    if ela_mean >= config.ela_high_mean:
        findings.append(_risk("ela_high", "Inconsistência forte de recompressão", 30,
                              f"ELA médio: {ela_mean:.2f}; percentil 95: {ela_p95:.2f}.", "alto"))
    elif ela_mean >= config.ela_warn_mean:
        findings.append(_risk("ela_warn", "Inconsistência de recompressão", 15,
                              f"ELA médio: {ela_mean:.2f}; percentil 95: {ela_p95:.2f}.", "médio"))

    sha256 = hashlib.sha256(data).hexdigest()
    phash = perceptual_hash(rgb)
    nearest_duplicate = None
    if known_hashes:
        distances = [(candidate, hamming_distance(phash, candidate)) for candidate in known_hashes]
        nearest_duplicate = min(distances, key=lambda item: item[1])
        if nearest_duplicate[1] <= config.duplicate_hamming_distance:
            findings.append(_risk("near_duplicate", "Imagem igual ou muito semelhante a caso anterior", 35,
                                  f"Distância perceptual: {nearest_duplicate[1]}.", "alto"))

    ocr_text, ocr_warning = _ocr(rgb)
    text_findings, field_matches = _text_checks(ocr_text, expected)
    findings.extend(text_findings)

    raw_score = sum(item["points"] for item in findings)
    score = min(100, raw_score)
    decision = "REVISAR" if score >= 35 else "ATENÇÃO" if score >= 15 else "BAIXO RISCO"
    return {
        "analysis_version": "0.1.0",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "sha256": sha256,
        "perceptual_hash": phash,
        "dimensions": {"width": width, "height": height},
        "format": image.format or "desconhecido",
        "blur_variance": round(blur_variance, 2),
        "ela": {"mean": round(ela_mean, 3), "p95": round(ela_p95, 3), "quality": config.ela_quality},
        "exif": exif,
        "ocr_text": ocr_text,
        "ocr_warning": ocr_warning,
        "field_matches": field_matches,
        "nearest_duplicate": ({"hash": nearest_duplicate[0], "distance": nearest_duplicate[1]}
                              if nearest_duplicate else None),
        "risk_score": score,
        "decision": decision,
        "findings": findings,
        "config": asdict(config),
        "disclaimer": "Indicadores de triagem; o resultado não comprova fraude e deve apoiar revisão humana.",
        "_ela_image": ela_render,
    }

