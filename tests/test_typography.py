import pytest

from conftest import fake_tesseract, receipt_words, text_line
from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.context import ImageContext
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, png_bytes, render_receipt
from fraud_detector.signals.typography import TypographySignal, comparable, line_outliers


def run_with_words(lines):
    ctx = ImageContext.from_bytes(png_bytes(render_receipt(ReceiptSpec(seed=1), size=(600, 400))), "x.png")
    words = []
    for index, line in enumerate(lines):
        for word in line:
            words.append({**word, "conf": word.get("conf", 90), "line": (1, 1, index + 1)})
    ctx.shared["ocr_words"] = words
    return TypographySignal().run(ctx, AnalysisConfig())


def test_consistent_lines_do_not_alert():
    result = run_with_words(receipt_words())
    assert result.findings == []
    assert result.features["typography_checked"] is True
    assert result.features["typography_outliers"] == 0


def test_taller_word_in_a_line_is_flagged_with_its_box():
    lines = receipt_words()
    lines[4] = text_line(340, "VALOR", "R$", "1.250,00")
    lines[4].append({"text": "9.250,00", "left": 700, "top": 330, "width": 240, "height": 60})
    result = run_with_words(lines)
    assert len(result.findings) == 1
    item = result.findings[0]
    assert item["code"] == "typography_inconsistent" and item["points"] == 10
    assert item["region"] == {"x": 700, "y": 330, "w": 240, "h": 60}
    assert result.features["typography_max_dev"] >= 0.3


def test_baseline_shift_is_flagged():
    lines = receipt_words()
    lines[1] = text_line(160, "PEDIDO", "NUM", "12345")
    lines[1][2]["top"] = 178  # mesma altura, 18 px abaixo da linha de base dos vizinhos
    result = run_with_words(lines)
    assert [o["text"] for o in result.details["outliers"]] == ["12345"]
    assert result.details["outliers"][0]["baseline_dev"] >= 0.35


def test_lowercase_with_descenders_is_not_compared():
    lines = receipt_words()
    lines[3] = text_line(280, "DESTINATARIO", "JOAO", "CLAUDIO")
    lines[3].append({"text": "pagamento", "left": 900, "top": 280, "width": 270, "height": 52})
    assert run_with_words(lines).findings == []
    assert not comparable("pagamento") and comparable("Total") and comparable("12345") and not comparable("-")


def test_heterogeneous_line_is_inconclusive():
    lines = [text_line(100, "A1", "B2", "C3", height=40)]
    lines[0][0]["height"] = 80
    lines[0][1]["height"] = 20
    checked, outliers = line_outliers([{**w, "conf": 90} for w in lines[0]], AnalysisConfig())
    assert checked and outliers == []


def test_too_few_words_skips_the_check():
    result = run_with_words([text_line(100, "PEDIDO", "12345")])
    assert result.features["typography_checked"] is False and result.findings == []


def test_pipeline_uses_words_published_by_ocr(monkeypatch):
    lines = receipt_words()
    lines[4].append({"text": "9.250,00", "left": 700, "top": 330, "width": 240, "height": 60})
    fake_tesseract(monkeypatch, lines)
    result = analyze_image(jpeg_bytes(render_receipt(ReceiptSpec(seed=2)), 85), "recibo.jpg")
    item = next(f for f in result["findings"] if f["code"] == "typography_inconsistent")
    # as caixas do OCR vêm na escala 2x e voltam à escala da imagem antes da checagem
    assert item["region"] == {"x": 350, "y": 165, "w": 120, "h": 30}
    assert result["features"]["typography.typography_lines"] >= 3
