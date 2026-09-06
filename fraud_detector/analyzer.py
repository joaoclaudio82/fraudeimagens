"""Orquestração: decodifica a imagem uma vez, executa cada sinal isoladamente e consolida o relatório."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import AnalysisConfig
from .context import ImageContext
from .scoring import build_scorer
from .signals import Signal, SignalResult, default_signals
from .signals.duplicates import hamming_distance, perceptual_hash  # noqa: F401  (compatibilidade)
from .signals.recompression import ela_image  # noqa: F401  (compatibilidade)

ANALYSIS_VERSION = "0.2.0"
DISCLAIMER = "Indicadores de triagem; o resultado não comprova fraude e deve apoiar revisão humana."


def run_signals(ctx: ImageContext, config: AnalysisConfig, signals: list[Signal]) -> list[SignalResult]:
    results: list[SignalResult] = []
    for signal in signals:
        try:
            results.append(signal.run(ctx, config))
        except Exception as exc:  # um sinal quebrado não pode derrubar a triagem
            results.append(SignalResult(signal.name, error=f"{type(exc).__name__}: {exc}"))
    return results


def analyze_image(
    data: bytes,
    filename: str = "image",
    expected: dict[str, str] | None = None,
    known_hashes: Any = None,
    config: AnalysisConfig | None = None,
    signals: list[Signal] | None = None,
    scorer: Any = None,
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    scorer = scorer or build_scorer(config)
    ctx = ImageContext.from_bytes(data, filename, expected, known_hashes, config)
    results = run_signals(ctx, config, signals or default_signals(config))

    findings: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    details: dict[str, Any] = {}
    images: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for item in results:
        findings.extend(item.findings)
        features.update({f"{item.name}.{key}": value for key, value in item.features.items()})
        details[item.name] = item.details
        images.update({f"{item.name}.{key}": value for key, value in item.images.items()})
        if item.error:
            errors[item.name] = item.error

    scored = scorer.score(findings, features, config)
    ocr = details.get("ocr", {})
    return {
        "analysis_version": ANALYSIS_VERSION,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "sha256": features.get("duplicates.sha256"),
        "perceptual_hash": features.get("duplicates.perceptual_hash"),
        "dimensions": {"width": ctx.width, "height": ctx.height},
        "format": ctx.format,
        "blur_variance": features.get("quality.blur_variance"),
        "ela": details.get("recompression", {}).get("ela"),
        "exif": details.get("metadata", {}).get("exif", {}),
        "ocr_text": ocr.get("text", ""),
        "ocr_warning": ocr.get("warning"),
        "field_matches": ocr.get("field_matches", {}),
        "nearest_duplicate": details.get("duplicates", {}).get("nearest_duplicate"),
        "risk_score": scored.score,
        "decision": scored.decision,
        "score_model": scored.model,
        "score_explanation": scored.explanation,
        "score_details": scored.extra,
        "findings": findings,
        "features": features,
        "signals": details,
        "signal_errors": errors,
        "config": config.to_dict(),
        "disclaimer": DISCLAIMER,
        "_ela_image": images.get("recompression.ela"),
        "_images": images,
    }
