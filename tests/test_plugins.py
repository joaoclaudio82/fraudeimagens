import json
import types

import numpy as np
import pytest

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.context import ImageContext
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt
from fraud_detector.signals.deep import DeepDetectorSignal, DeepPrediction, load_predictor, mask_regions
from fraud_detector.signals.vlm import SCHEMA, VlmSignal


@pytest.fixture(scope="module")
def receipt_bytes():
    return jpeg_bytes(render_receipt(ReceiptSpec(seed=51)), 85)


# ----------------------------------------------------------------------------- detector profundo

def fake_predict_high(rgb):
    mask = np.zeros((90, 120), dtype=np.float32)
    mask[35:45, 10:70] = 0.95  # faixa na área da linha do valor (escala 10x)
    return DeepPrediction(score=0.9, mask=mask, model="fake-net", details={"version": "1"})


def fake_predict_low(rgb):
    return {"score": 0.1, "model": "fake-net"}


def test_deep_signal_is_inert_without_model(receipt_bytes):
    result = analyze_image(receipt_bytes, "a.jpg")
    assert result["features"]["deep.deep_enabled"] is False
    assert not [f for f in result["findings"] if f["code"].startswith("deep")]


def test_deep_signal_uses_configured_predictor(receipt_bytes):
    signals = [DeepDetectorSignal(predictor=fake_predict_high)]
    result = analyze_image(receipt_bytes, "a.jpg", signals=signals)
    item = next(f for f in result["findings"] if f["code"] == "deep_manipulation")
    assert item["points"] == 30 and "fake-net" in item["label"]
    region = item["region"]
    assert 90 <= region["x"] <= 110 and 340 <= region["y"] <= 360  # máscara 120x90 escalada para 1200x900
    assert result["features"]["deep.deep_score"] == 0.9 and result["features"]["deep.deep_regions"] == 1
    assert result["_images"]["deep.overlay"].size == (1200, 900)
    assert result["signals"]["deep"]["version"] == "1"

    low = analyze_image(receipt_bytes, "a.jpg", signals=[DeepDetectorSignal(predictor=fake_predict_low)])
    assert not [f for f in low["findings"] if f["code"].startswith("deep")]
    assert low["features"]["deep.deep_score"] == 0.1


def test_deep_predictor_is_loaded_from_spec(monkeypatch, receipt_bytes):
    import sys

    module = types.ModuleType("fake_deep_plugin")
    module.predict = fake_predict_high
    monkeypatch.setitem(sys.modules, "fake_deep_plugin", module)
    assert load_predictor("fake_deep_plugin:predict") is fake_predict_high
    assert load_predictor(None) is None
    monkeypatch.setenv("IMAGEGUARD_DEEP_MODEL", "fake_deep_plugin")
    assert load_predictor(None) is fake_predict_high

    result = analyze_image(receipt_bytes, "a.jpg", config=AnalysisConfig(deep_model="fake_deep_plugin:predict"))
    assert any(f["code"] == "deep_manipulation" for f in result["findings"])


def test_mask_regions_ignores_tiny_blobs():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:12, 10:12] = 1.0
    assert mask_regions(mask, (1000, 1000), 0.5, min_area_fraction=0.01) == []
    mask[50:90, 20:80] = 1.0
    boxes = mask_regions(mask, (1000, 1000), 0.5, min_area_fraction=0.01)
    assert len(boxes) == 1 and boxes[0]["y"] == 500


# ----------------------------------------------------------------------------- modelo de visão

class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text=json.dumps(self.payload))
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[block],
                                     stop_details=types.SimpleNamespace(category="other"))


def fake_client(payload, stop_reason="end_turn"):
    messages = FakeMessages(payload, stop_reason)
    return types.SimpleNamespace(messages=messages)


def extracted(**overrides):
    base = {"document_type": "comprovante de entrega", "order_id": "12345", "date": "04/09/2026",
            "recipient": "JOAO CLAUDIO", "value": "R$ 1.250,00", "code": "ABC-7781", "has_signature": True,
            "has_stamp": False, "legibility": 0.9, "edit_signs": [], "notes": ""}
    base.update(overrides)
    return base


def test_vlm_is_disabled_by_default(receipt_bytes):
    result = analyze_image(receipt_bytes, "a.jpg")
    assert result["features"]["vlm.vlm_enabled"] is False
    assert result["signal_errors"] == {}


def test_vlm_matches_fields_and_sends_schema(receipt_bytes):
    client = fake_client(extracted())
    config = AnalysisConfig(vlm_enabled=True)
    signals = [VlmSignal(client=client)]
    expected = {"pedido": "12345", "data": "2026-09-04", "valor": "R$ 1250,00", "destinatario": "João Cláudio"}
    result = analyze_image(receipt_bytes, "a.jpg", expected=expected, config=config, signals=signals)
    assert result["features"]["vlm.vlm_fields_matched"] == 4 and result["features"]["vlm.vlm_fields_conflict"] == 0
    assert not [f for f in result["findings"] if f["code"].startswith("vlm_")]

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5" and call["output_config"]["format"]["schema"] == SCHEMA
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text"


def test_vlm_reports_conflicts_edit_signs_and_signature(receipt_bytes):
    client = fake_client(extracted(value="R$ 9.250,00", edit_signs=["fonte diferente no valor"], has_signature=False))
    config = AnalysisConfig(vlm_enabled=True, vlm_require_signature=True)
    result = analyze_image(receipt_bytes, "a.jpg", expected={"valor": "R$ 1.250,00", "pedido": "12345"},
                           config=config, signals=[VlmSignal(client=client)])
    codes = {f["code"]: f for f in result["findings"]}
    assert codes["vlm_field_valor"]["points"] == 18 and "9.250,00" in codes["vlm_field_valor"]["evidence"]
    assert codes["vlm_edit_signs"]["points"] == 12 and "fonte" in codes["vlm_edit_signs"]["evidence"]
    assert codes["vlm_no_signature"]["points"] == 5
    assert result["features"]["vlm.vlm_edit_signs"] == 1


def test_vlm_handles_refusal_and_errors(receipt_bytes):
    config = AnalysisConfig(vlm_enabled=True)
    refused = analyze_image(receipt_bytes, "a.jpg", config=config,
                            signals=[VlmSignal(client=fake_client(extracted(), stop_reason="refusal"))])
    item = next(f for f in refused["findings"] if f["code"] == "vlm_unavailable")
    assert item["points"] == 0 and "recusou" in item["evidence"]

    class Broken:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise ConnectionError("sem rede")

    failed = analyze_image(receipt_bytes, "a.jpg", config=config, signals=[VlmSignal(client=Broken())])
    assert failed["signal_errors"] == {}
    assert "ConnectionError" in failed["signals"]["vlm"]["warning"]
