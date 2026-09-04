import numpy as np
import pytest
from PIL import Image

from conftest import fake_tesseract, receipt_words
from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt
from fraud_detector.signals.ocr import estimate_skew, preprocess_for_ocr, text_from_words


@pytest.fixture
def image_data():
    return jpeg_bytes(render_receipt(ReceiptSpec(seed=5)), 85)


def test_fields_match_semantically_with_ocr_output(monkeypatch, image_data):
    fake_tesseract(monkeypatch, receipt_words())
    expected = {"pedido": "12345", "data": "2026-09-04", "destinatário": "João Cláudio",
                "valor": "R$ 1250,00", "código": "ABC-7781"}
    result = analyze_image(image_data, "recibo.jpg", expected=expected)
    assert result["ocr_warning"] is None
    assert all(result["field_matches"].values()), result["signals"]["ocr"]["field_details"]
    assert not [f for f in result["findings"] if f["code"].startswith("field_")]
    assert result["features"]["ocr.ocr_words"] == 15
    assert result["features"]["ocr.ocr_mean_conf"] == 90.0


def test_conflicting_value_is_flagged_strongly(monkeypatch, image_data):
    fake_tesseract(monkeypatch, receipt_words())
    result = analyze_image(image_data, "recibo.jpg", expected={"valor": "R$ 9.250,00", "data": "05/09/2026"})
    codes = {f["code"]: f for f in result["findings"]}
    assert codes["field_valor"]["points"] == 18 and codes["field_valor"]["severity"] == "alto"
    assert "1250.00" in codes["field_valor"]["evidence"]
    assert codes["field_data"]["points"] == 18
    assert result["features"]["ocr.fields_conflict"] == 2


def test_missing_field_is_a_weak_signal(monkeypatch, image_data):
    fake_tesseract(monkeypatch, receipt_words())
    result = analyze_image(image_data, "recibo.jpg", expected={"transportadora": "Loggi Express"})
    item = next(f for f in result["findings"] if f["code"] == "field_transportadora")
    assert item["points"] == 8 and item["severity"] == "baixo"


def test_unavailable_ocr_does_not_penalize_fields(monkeypatch, image_data):
    fake_tesseract(monkeypatch, receipt_words(), fail_all=True)
    result = analyze_image(image_data, "recibo.jpg", expected={"pedido": "12345", "valor": "R$ 1,00"})
    assert result["ocr_warning"]
    assert result["field_matches"] == {}
    assert not [f for f in result["findings"] if f["code"].startswith("field_")]
    assert any(f["code"] == "ocr_unavailable" and f["points"] == 0 for f in result["findings"])


def test_falls_back_to_image_to_string(monkeypatch, image_data):
    calls = fake_tesseract(monkeypatch, receipt_words(), fail_data=True)
    result = analyze_image(image_data, "recibo.jpg", expected={"pedido": "12345"})
    assert calls["string"] >= 1
    assert result["field_matches"] == {"pedido": True}


def test_empty_ocr_text_is_reported(monkeypatch, image_data):
    fake_tesseract(monkeypatch, [])
    result = analyze_image(image_data, "recibo.jpg", expected={"pedido": "12345"})
    assert any(f["code"] == "ocr_empty" for f in result["findings"])
    assert result["field_matches"] == {}


def test_word_boxes_are_scaled_back_and_shared(monkeypatch, image_data):
    fake_tesseract(monkeypatch, receipt_words())
    from fraud_detector.context import ImageContext
    from fraud_detector.signals.ocr import OcrSignal

    ctx = ImageContext.from_bytes(image_data, "recibo.jpg")
    OcrSignal().run(ctx, AnalysisConfig())
    words = ctx.shared["ocr_words"]
    # a imagem tem 1200 px de largura, então foi ampliada 2x e as caixas voltam à escala original
    assert words[0]["left"] == 50 and words[0]["top"] == 50


def test_preprocessing_upscales_small_images_and_estimates_skew():
    receipt = render_receipt(ReceiptSpec(seed=7))
    rotated = receipt.rotate(3, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(200, 200, 200))
    prepared, info = preprocess_for_ocr(np.asarray(rotated), AnalysisConfig())
    assert info["scale"] == 2.0
    assert prepared.size == (2400, 1800)
    assert abs(abs(info["angle"]) - 3) <= 1.5

    flat, info = preprocess_for_ocr(np.asarray(receipt), AnalysisConfig(ocr_preprocess=False))
    assert info == {"scale": 1.0, "angle": 0.0} and flat.size == receipt.size


def test_text_is_rebuilt_from_word_lines():
    words = [
        {"text": "VALOR", "left": 10, "top": 100, "width": 50, "height": 20, "line": (1, 1, 2)},
        {"text": "PEDIDO", "left": 10, "top": 40, "width": 50, "height": 20, "line": (1, 1, 1)},
        {"text": "12345", "left": 80, "top": 40, "width": 50, "height": 20, "line": (1, 1, 1)},
    ]
    assert text_from_words(words) == "PEDIDO 12345\nVALOR"


def test_estimate_skew_on_blank_image_is_zero():
    assert estimate_skew(np.zeros((100, 100), dtype=np.uint8), 15.0) == 0.0
