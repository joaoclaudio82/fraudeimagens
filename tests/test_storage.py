from datetime import datetime, timedelta, timezone

import pytest

from fraud_detector import analyze_image
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt, whatsapp_like
from fraud_detector.storage import AnalysisStore, strip_report


@pytest.fixture
def receipt_bytes():
    return jpeg_bytes(render_receipt(ReceiptSpec(seed=31)), 85)


def test_save_get_and_duplicate_detection_via_store(receipt_bytes):
    store = AnalysisStore(retention_days=30)
    first = analyze_image(receipt_bytes, "a.jpg", {"pedido": "111"}, store.hashes)
    first_id = store.save(first, reference="PED-111")
    saved = store.get(first_id)
    assert saved["reference"] == "PED-111" and saved["status"] == "pending"
    assert "_images" not in saved["report"] and saved["report"]["risk_score"] == first["risk_score"]
    assert store.hashes.count() == 1

    again = analyze_image(whatsapp_like(receipt_bytes), "b.jpg", {"pedido": "222"}, store.hashes)
    assert any(f["code"] == "near_duplicate" and "PED-111" in f["evidence"] for f in again["findings"])
    assert store.get(9999) is None


def test_queue_review_labels_and_stats(receipt_bytes):
    store = AnalysisStore()
    low = analyze_image(receipt_bytes, "low.jpg")
    low["risk_score"], low["decision"] = 5, "BAIXO RISCO"
    high = dict(low, risk_score=80, decision="REVISAR", filename="high.jpg")
    mid = dict(low, risk_score=20, decision="ATENÇÃO", filename="mid.jpg")
    ids = [store.save(item, register_hashes=False) for item in (low, high, mid)]

    queue = store.queue()
    assert [item["filename"] for item in queue] == ["high.jpg", "mid.jpg"]
    assert [item["filename"] for item in store.queue(min_decision="REVISAR")] == ["high.jpg"]
    assert store.queue(limit=1)[0]["score"] == 80

    store.review(ids[1], "legitimate", reviewer="ana", note="conferido com a transportadora")
    store.review(ids[2], "confirmed_fraud", reviewer="ana")
    assert [item["filename"] for item in store.queue()] == []
    assert store.get(ids[1])["reviewer"] == "ana" and store.get(ids[1])["note"]

    rows = store.labeled_rows()
    assert [(row["file"], row["label"]) for row in rows] == [("high.jpg", 0), ("mid.jpg", 1)]
    assert "quality.blur_variance" in rows[0]["features"]

    stats = store.stats()
    assert stats["analyses"] == 3 and stats["reviewed"] == 2 and stats["confirmed_fraud"] == 1
    assert stats["reversal_rate"] == 1.0  # o único REVISAR revisado era legítimo
    assert stats["by_review_status"]["pending"] == 1

    with pytest.raises(ValueError):
        store.review(ids[0], "qualquer")
    with pytest.raises(KeyError):
        store.review(12345, "legitimate")

    listing = store.list(limit=2)
    assert [item["filename"] for item in listing] == ["mid.jpg", "high.jpg"]
    assert store.list(decision="REVISAR")[0]["filename"] == "high.jpg"


def test_retention_purge_and_ocr_text_policy(receipt_bytes, tmp_path):
    store = AnalysisStore(tmp_path / "db.sqlite", retention_days=1, store_ocr_text=False)
    report = analyze_image(receipt_bytes, "a.jpg")
    report["ocr_text"] = "JOAO CLAUDIO RUA X 123"
    analysis_id = store.save(report)
    assert store.get(analysis_id)["report"]["ocr_text"] == ""
    assert store.purge_expired() == 0
    assert store.purge_expired(now=datetime.now(timezone.utc) + timedelta(days=2)) == 1
    assert store.get(analysis_id) is None and store.stats()["analyses"] == 0
    store.close()

    reopened = AnalysisStore(tmp_path / "db.sqlite")
    assert reopened.stats()["hashes"] == 1  # hashes seguem disponíveis para comparação
    reopened.close()


def test_strip_report_keeps_serializable_fields(receipt_bytes):
    report = analyze_image(receipt_bytes, "a.jpg")
    cleaned = strip_report(report)
    assert "_images" not in cleaned and "_ela_image" not in cleaned and cleaned["findings"] == report["findings"]


def test_queue_filters_before_limit_and_preserves_tie_order():
    store = AnalysisStore()
    def save(score, decision):
        return store.save({'risk_score': score, 'decision': decision}, register_hashes=False)
    save(99, 'BAIXO RISCO')
    first = save(80, 'REVISAR')
    second = save(80, 'REVISAR')
    reviewed = save(90, 'REVISAR')
    store.review(reviewed, 'legitimate')
    assert [row['id'] for row in store.queue(limit=1)] == [first]
    assert [row['id'] for row in store.queue(limit=2)] == [first, second]
    assert store.queue(limit=0) == []
    assert store.queue(limit=-1) == []
    with pytest.raises(ValueError):
        store.queue(min_decision='typo')
    store.close()
