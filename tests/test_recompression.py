import pytest

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.evaluation.synth import (
    ReceiptSpec, double_compressed, forge_value_edit, jpeg_bytes, line_region, png_bytes, render_receipt,
)
from fraud_detector.jpegutil import estimate_quality, jpeg_summary, quantization_tables
from fraud_detector.signals.recompression import jpeg_ghost


@pytest.fixture(scope="module")
def receipt():
    return render_receipt(ReceiptSpec(seed=3))


def covered_fraction(box, target):
    """Fração da região alvo coberta pela caixa detectada."""
    x0 = max(box["x"], target["x"])
    y0 = max(box["y"], target["y"])
    x1 = min(box["x"] + box["w"], target["x"] + target["w"])
    y1 = min(box["y"] + box["h"], target["y"] + target["h"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (target["w"] * target["h"])


def ghost_findings(result):
    return [item for item in result["findings"] if item["code"] == "ghost_local"]


def test_intact_double_compressed_receipt_has_no_local_ghost(receipt):
    result = analyze_image(double_compressed(receipt, 75, 90), "integra.jpg")
    assert ghost_findings(result) == []
    assert result["features"]["recompression.ghost_found"] is False


def test_edited_value_line_is_localized(receipt):
    original = jpeg_bytes(receipt, 75)
    forged, edited = forge_value_edit(original, quality=90)
    result = analyze_image(forged, "forjada.jpg")

    found = ghost_findings(result)
    assert found, result["signals"]["recompression"]["ghost"]
    box = found[0]["region"]
    assert covered_fraction(box, edited) >= 0.5
    assert box["w"] * box["h"] <= 5 * edited["w"] * edited["h"]
    assert result["signals"]["recompression"]["ghost"]["ghost_quality"] == 75
    assert result["decision"] != "BAIXO RISCO"


def test_png_without_jpeg_history_is_quiet(receipt):
    result = analyze_image(png_bytes(receipt), "captura.png")
    assert ghost_findings(result) == []
    assert result["features"]["recompression.ghost_found"] is False


def test_ghost_overlay_and_ela_images_are_produced(receipt):
    result = analyze_image(jpeg_bytes(receipt, 80), "foto.jpg")
    assert result["_ela_image"].size == receipt.size
    assert result["_images"]["recompression.ghost_overlay"].size == receipt.size


def test_ghost_can_be_disabled(receipt):
    result = analyze_image(jpeg_bytes(receipt, 80), "foto.jpg", config=AnalysisConfig(ghost_enabled=False))
    assert "recompression.ghost_median_dip" not in result["features"]
    assert result["features"]["recompression.ghost_found"] is False


def test_jpeg_quality_is_estimated_from_quantization_tables(receipt):
    from PIL import Image
    import io

    for quality in (60, 75, 90):
        image = Image.open(io.BytesIO(jpeg_bytes(receipt, quality)))
        tables = quantization_tables(image)
        assert abs(estimate_quality(tables[0]) - quality) <= 1
        summary = jpeg_summary(image)
        assert summary["is_jpeg"] and summary["standard_tables"] is True
        assert summary["estimated_quality"] == quality
    assert jpeg_summary(Image.open(io.BytesIO(png_bytes(receipt)))) == {"is_jpeg": False}


def test_ghost_function_reports_regions_in_pixels(receipt):
    import numpy as np
    import cv2

    forged, edited = forge_value_edit(jpeg_bytes(receipt, 75), quality=90)
    from fraud_detector.evaluation.synth import open_rgb

    rgb = open_rgb(forged)
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    analysis = jpeg_ghost(rgb, gray, AnalysisConfig(), last_quality=90)
    assert analysis.found
    assert analysis.regions[0]["x"] % analysis.block == 0
    assert covered_fraction(analysis.regions[0], edited) >= 0.5
    assert line_region("valor")["y"] == edited["y"]
