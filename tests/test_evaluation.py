import json

import pytest

from fraud_detector import AnalysisConfig
from fraud_detector.evaluation.harness import (
    ItemResult, build_report, load_results, region_hit, report_markdown, run_dataset, save_results,
)
from fraud_detector.evaluation.metrics import best_threshold, binary_metrics, pr_auc, pr_curve, roc_auc
from fraud_detector.evaluation.synth import FORGED_KINDS, INTACT_ORIGINS, generate_dataset


def test_metrics_on_perfect_and_random_predictors():
    y = [0, 0, 1, 1, 0, 1]
    assert pr_auc(y, [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]) == 1.0
    assert roc_auc(y, [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]) == 1.0
    assert roc_auc(y, [0.5] * 6) == 0.5
    assert pr_auc([0, 0, 0], [1, 2, 3]) == 0.0
    assert pr_curve([1, 0, 1], [0.9, 0.9, 0.1])[0] == (0.5, 0.5, 0.9)

    m = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["fpr"] == 0.5

    best = best_threshold([0, 0, 1, 1], [10, 20, 30, 40], cost_fp=1, cost_fn=5)
    assert best["threshold"] == 30 and best["fp"] == 0 and best["fn"] == 0


def test_region_hit_by_center_and_iou():
    region = {"x": 100, "y": 100, "w": 200, "h": 50}
    assert region_hit([{"region": {"x": 150, "y": 110, "w": 40, "h": 20}}], region) is True
    assert region_hit([{"region": {"x": 0, "y": 90, "w": 220, "h": 60}}], region) is True   # IoU
    assert region_hit([{"region": {"x": 900, "y": 900, "w": 10, "h": 10}}], region) is False
    assert region_hit([{"code": "x"}], region) is False
    assert region_hit([], None) is None


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("synth")
    items = generate_dataset(folder, n_intact=len(INTACT_ORIGINS), n_forged=len(FORGED_KINDS), seed=3)
    return folder, items


def test_generated_dataset_is_labeled_and_diverse(dataset):
    folder, items = dataset
    assert len(items) == len(INTACT_ORIGINS) + len(FORGED_KINDS)
    lines = (folder / "labels.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(items)
    assert {json.loads(line)["origin"] for line in lines[:len(INTACT_ORIGINS)]} == set(INTACT_ORIGINS)
    assert {item["kind"] for item in items if item["label"] == 1} == set(FORGED_KINDS)
    for item in items:
        assert (folder / item["file"]).stat().st_size > 1000
        assert set(item["expected"]) == {"pedido", "data", "destinatario", "valor", "codigo"}
    edited = [item for item in items if item["kind"] in ("value_edit", "copy_move", "splice")]
    assert all(item["region"] for item in edited)


def test_harness_runs_dataset_and_builds_report(dataset, tmp_path):
    folder, items = dataset
    config = AnalysisConfig()
    seen = []
    results = run_dataset(folder, config, progress=lambda done, total, item: seen.append((done, total)))
    assert len(results) == len(items) and seen[-1] == (len(items), len(items))
    assert all(r.errors == {} for r in results), [r.errors for r in results]

    report = build_report(results, config)
    assert report["n"] == len(items) and report["positives"] == len(FORGED_KINDS)
    assert 0.0 <= report["combined"]["pr_auc"] <= 1.0 and 0.0 <= report["combined"]["roc_auc"] <= 1.0
    assert set(report["per_kind"]) == set(FORGED_KINDS)
    assert set(report["per_origin"]) == set(INTACT_ORIGINS)
    assert "ghost_local" in report["per_signal"] or "copy_move" in report["per_signal"]
    # a foto reaproveitada só é pega porque o harness registra os hashes na ordem, como em produção
    assert "near_duplicate" in [code for r in results if r.kind == "reuse" for code in r.codes]

    markdown = report_markdown(report, "Teste")
    assert markdown.startswith("# Teste") and "| ghost_local" in markdown or "| copy_move" in markdown

    path = tmp_path / "results.jsonl"
    save_results(results, path)
    loaded = load_results(path)
    assert [r.file for r in loaded] == [r.file for r in results]
    assert isinstance(loaded[0], ItemResult) and loaded[0].features == results[0].features
