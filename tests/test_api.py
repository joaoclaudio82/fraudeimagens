import base64
import json

import pytest
from fastapi.testclient import TestClient

from fraud_detector import AnalysisConfig
from fraud_detector.api import create_app
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt, whatsapp_like, with_exif
from fraud_detector.storage import AnalysisStore


@pytest.fixture
def client():
    app = create_app(store=AnalysisStore(), config=AnalysisConfig())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def receipt_bytes():
    return jpeg_bytes(render_receipt(ReceiptSpec(seed=41)), 85)


def test_health(client):
    body = client.get("/stats").json()
    assert body["status"] == "ok" and "ocr_available" in body and body["analyses"] == 0


def test_analyze_register_and_review_flow(client, receipt_bytes):
    response = client.post("/analyze", files={"file": ("a.jpg", receipt_bytes, "image/jpeg")},
                           data={"expected": json.dumps({"pedido": "12345"}), "reference": "PED-1"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["analysis_id"] == 1 and 0 <= body["risk_score"] <= 100 and "images" not in body
    assert body["field_matches"] in ({}, {"pedido": True}, {"pedido": False})

    resent = client.post("/analyze", files={"file": ("b.jpg", whatsapp_like(receipt_bytes), "image/jpeg")},
                         data={"pedido": "99999", "include_images": "true"}).json()
    assert any(f["code"] == "near_duplicate" and "PED-1" in f["evidence"] for f in resent["findings"])
    assert set(resent["images"]) >= {"recompression.ela", "recompression.ghost_overlay", "copy_move.overlay"}
    assert base64.b64decode(resent["images"]["recompression.ela"])[:8] == b"\x89PNG\r\n\x1a\n"

    listing = client.get("/analyses").json()
    assert [item["id"] for item in listing] == [2, 1]
    assert client.get("/analyses/1").json()["reference"] == "PED-1"
    assert client.get("/analyses/77").status_code == 404

    queue = client.get("/review-queue", params={"min_decision": "BAIXO RISCO"}).json()
    assert [item["id"] for item in queue][0] == 2  # o reenvio tem score maior

    assert client.post("/analyses/2/review", json={"status": "confirmed_fraude"}).status_code == 400
    assert client.post("/analyses/77/review", json={"status": "legitimate"}).status_code == 404
    done = client.post("/analyses/2/review", json={"status": "confirmed_fraud", "reviewer": "ana", "note": "reuso"})
    assert done.json() == {"analysis_id": 2, "status": "confirmed_fraud"}

    labels = client.get("/export/labels").json()
    assert len(labels) == 1 and labels[0]["label"] == 1 and "features" in labels[0]
    assert client.post("/maintenance/purge").json() == {"removed": 0}
    assert client.get("/stats").json()["confirmed_fraud"] == 1


def test_analyze_without_register_and_bad_inputs(client, receipt_bytes):
    body = client.post("/analyze", files={"file": ("a.jpg", receipt_bytes, "image/jpeg")}, data={"persist": "false"}).json()
    assert "analysis_id" not in body
    assert client.get("/analyses").json() == []
    assert client.post("/analyze", files={"file": ("x.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/analyze", files={"file": ("x.jpg", b"nao e imagem", "image/jpeg")}).status_code == 422
    assert client.post("/analyze", files={"file": ("a.jpg", receipt_bytes, "image/jpeg")},
                       data={"expected": "{nao json"}).status_code == 400


def test_editor_metadata_reaches_the_queue(client, receipt_bytes):
    edited = with_exif(receipt_bytes, software="Adobe Photoshop 25.0")
    body = client.post("/analyze", files={"file": ("e.jpg", edited, "image/jpeg")}).json()
    assert any(f["code"] == "editing_software" for f in body["findings"])
    queue = client.get("/review-queue").json()
    assert queue and queue[0]["findings"][0]["code"]


@pytest.fixture(autouse=True)
def local_development_auth(monkeypatch):
    monkeypatch.setenv("IMAGEGUARD_AUTH_MODE", "disabled")
