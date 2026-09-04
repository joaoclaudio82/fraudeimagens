import json

from fraud_detector.cli import main, parse_expected
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt
from fraud_detector.storage import AnalysisStore


def test_parse_expected():
    assert parse_expected(["pedido=123", "data = 04/09/2026", "invalido"]) == {"pedido": "123", "data": "04/09/2026"}
    assert parse_expected(None) == {}


def test_analyze_command_writes_jsonl_and_db(tmp_path, capsys):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for seed in (1, 2):
        (folder / f"r{seed}.jpg").write_bytes(jpeg_bytes(render_receipt(ReceiptSpec(seed=seed)), 85))
    (folder / "ignorado.txt").write_text("x")
    out = tmp_path / "out.jsonl"
    db = tmp_path / "db.sqlite"

    code = main(["analyze", str(folder), "--expected", "pedido=12345", "--db", str(db), "--out", str(out),
                 "--reference", "LOTE-1"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "r1.jpg" in printed and "r2.jpg" in printed
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["filename"] for row in rows] == ["r1.jpg", "r2.jpg"] and all("analysis_id" in row for row in rows)

    store = AnalysisStore(db)
    assert store.stats()["analyses"] == 2 and store.stats()["hashes"] == 2
    assert store.list()[0]["reference"] == "LOTE-1"
    store.close()

    assert main(["stats", "--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["analyses"] == 2
    assert main(["purge", "--db", str(db)]) == 0
    assert "0 análise(s)" in capsys.readouterr().out


def test_analyze_command_without_images(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nada")]) == 1
