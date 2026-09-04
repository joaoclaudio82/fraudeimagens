import io
import json

from PIL import Image, ImageDraw, ImageFilter

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.analyzer import hamming_distance, perceptual_hash
from fraud_detector.signals import default_signals
from fraud_detector.signals.base import SignalResult, finding


def image_bytes(blur=False):
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 90, 820, 510), outline="black", width=4)
    draw.text((120, 150), "PEDIDO 12345", fill="black")
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(8))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_hash_is_stable():
    data = image_bytes()
    image = Image.open(io.BytesIO(data))
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    assert hamming_distance(perceptual_hash(image), perceptual_hash(image)) == 0


def test_analysis_has_auditable_output():
    result = analyze_image(image_bytes(), "proof.png", expected={"pedido": "12345"})
    assert len(result["sha256"]) == 64
    assert 0 <= result["risk_score"] <= 100
    assert result["decision"] in {"BAIXO RISCO", "ATENÇÃO", "REVISAR"}
    assert result["disclaimer"]
    assert result.pop("_ela_image").size == (900, 600)
    result.pop("_images")
    # Tudo que sobra precisa ser serializável para o registro de auditoria.
    json.dumps(result, ensure_ascii=False)
    assert result["signal_errors"] == {}
    assert "quality.blur_variance" in result["features"]


def test_blurred_image_is_flagged():
    result = analyze_image(image_bytes(blur=True), "blurred.png")
    assert any(item["code"] == "blur" for item in result["findings"])


def test_decision_thresholds_come_from_config():
    config = AnalysisConfig(review_threshold=5, attention_threshold=1)
    result = analyze_image(image_bytes(blur=True), "blurred.png", config=config)
    assert result["risk_score"] >= 10
    assert result["decision"] == "REVISAR"
    assert result["config"]["review_threshold"] == 5


def test_failing_signal_does_not_break_analysis():
    class Broken:
        name = "broken"

        def run(self, ctx, config):
            raise RuntimeError("boom")

    class Custom:
        name = "custom"

        def run(self, ctx, config):
            result = SignalResult(self.name)
            result.add(finding("custom_flag", "Sinal customizado", 20, "teste", "médio"))
            return result

    signals = default_signals() + [Broken(), Custom()]
    result = analyze_image(image_bytes(), "proof.png", signals=signals)
    assert result["signal_errors"] == {"broken": "RuntimeError: boom"}
    assert any(item["code"] == "custom_flag" for item in result["findings"])
    assert result["risk_score"] >= 20
