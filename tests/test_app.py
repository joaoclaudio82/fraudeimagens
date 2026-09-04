"""Executa a página Streamlit sem servidor (AppTest). O upload não é simulável; cobre o fluxo sem imagem."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fraud_detector.storage import AnalysisStore  # noqa: E402


def test_page_renders_with_empty_store(tmp_path, monkeypatch):
    db = tmp_path / "ui.sqlite"
    monkeypatch.setenv("IMAGEGUARD_DB", str(db))
    monkeypatch.setenv("IMAGEGUARD_SCORING_MODEL", str(tmp_path / "nao_existe.json"))

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert "ImageGuard" in at.title[0].value
    assert any("Nenhuma análise pendente" in item.value for item in at.info)
    assert [metric.label for metric in at.metric][:2] == ["Análises registradas", "Revisadas"]
    assert at.metric[0].value == "0"
    assert db.exists()


def test_review_queue_lists_pending_items(tmp_path, monkeypatch):
    db = tmp_path / "ui.sqlite"
    store = AnalysisStore(db)
    store.save({"filename": "suspeita.jpg", "sha256": "", "risk_score": 60, "decision": "REVISAR",
                "score_model": "additive", "findings": [{"code": "copy_move", "label": "x", "points": 30, "severity": "alto"}],
                "features": {}}, reference="PED-9", register_hashes=False)
    store.close()
    monkeypatch.setenv("IMAGEGUARD_DB", str(db))

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("1 pendente" in item.value for item in at.subheader)
    assert any("PED-9" in item.label for item in at.expander)
    assert at.metric[0].value == "1"
