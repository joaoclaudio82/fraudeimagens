import json

import numpy as np
import pytest

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, render_receipt
from fraud_detector.evaluation.train import collect_feature_names, save_model, train_model
from fraud_detector.scoring import AdditiveScorer, LogisticScorer, MaxScorer, build_scorer, decide, feature_value


def synthetic_rows(n=200, seed=0):
    """Features fictícias: 'a' e a categoria 'kind' separam as classes; 'noise' não."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        label = int(rng.random() < 0.4)
        rows.append({
            "label": label,
            "features": {
                "sig.a": float(rng.normal(2.0 if label else 0.0, 1.0)),
                "sig.flag": bool(rng.random() < (0.7 if label else 0.1)),
                "sig.noise": float(rng.normal()),
                "meta.kind": str(rng.choice(["editor", "firmware"], p=[0.8, 0.2] if label else [0.1, 0.9])),
                "duplicates.sha256": "deadbeef" * 8,
                "sig.missing": float(rng.normal()) if rng.random() < 0.5 else None,
            },
        })
    for row in rows:
        if row["features"]["sig.missing"] is None:
            del row["features"]["sig.missing"]
    return rows


def test_feature_names_exclude_hashes_and_encode_categories():
    names = collect_feature_names(synthetic_rows(50))
    assert "duplicates.sha256" not in names
    assert {"sig.a", "sig.flag", "sig.noise", "sig.missing", "meta.kind=editor", "meta.kind=firmware"} <= set(names)


def test_training_produces_useful_and_explainable_model(tmp_path):
    rows = synthetic_rows()
    model = train_model(rows, holdout=0.25, seed=1)
    assert model["model"] == "logistic" and model["n_train"] == 200
    assert model["metrics"]["holdout"]["roc_auc"] >= 0.9
    assert model["metrics"]["train"]["pr_auc"] >= 0.85
    top = [item["feature"] for item in model["top_features"][:3]]
    assert "sig.a" in top
    coef = dict(zip(model["features"], model["coef"]))
    assert coef["meta.kind=editor"] > 0 > coef["meta.kind=firmware"]
    assert abs(coef["sig.noise"]) < abs(coef["sig.a"])

    path = tmp_path / "model.json"
    save_model(model, path)
    scorer = LogisticScorer.from_file(path)
    config = AnalysisConfig()
    high = scorer.score([], {"sig.a": 3.0, "sig.flag": True, "meta.kind": "editor"}, config)
    low = scorer.score([], {"sig.a": -1.0, "sig.flag": False, "meta.kind": "firmware"}, config)
    assert high.score > 70 > 30 > low.score
    assert high.explanation[0]["source"] in {"sig.a", "sig.flag", "meta.kind=editor"}
    assert high.extra["probability"] > 0.7
    # feature ausente é imputada pela média e categoria desconhecida vale zero
    middle = scorer.score([], {"meta.kind": "outra"}, config)
    assert 0 <= middle.score <= 100


def test_training_rejects_degenerate_bases():
    rows = synthetic_rows(20)
    for row in rows:
        row["label"] = 0
    with pytest.raises(ValueError):
        train_model(rows)
    with pytest.raises(ValueError):
        train_model(rows[:4])


def test_feature_value_conversions():
    features = {"x.n": 3, "x.b": True, "x.s": "abc", "x.f": 1.5}
    assert feature_value(features, "x.n") == 3.0 and feature_value(features, "x.b") == 1.0
    assert feature_value(features, "x.s") is None and feature_value(features, "x.f") == 1.5
    assert feature_value(features, "x.s=abc") == 1.0 and feature_value(features, "x.s=zzz") == 0.0
    assert feature_value(features, "y.q=abc") is None and feature_value(features, "y.q") is None


def test_decision_and_additive():
    config = AnalysisConfig(review_threshold=40, attention_threshold=20)
    assert decide(40, config) == "REVISAR" and decide(20, config) == "ATENÇÃO" and decide(19, config) == "BAIXO RISCO"
    result = AdditiveScorer().score([{"code": "a", "label": "A", "points": 70}, {"code": "b", "label": "B", "points": 50}], {}, config)
    assert result.score == 100 and result.explanation[0]["source"] == "a"


def test_analyzer_uses_configured_model_and_max_mode(tmp_path):
    data = jpeg_bytes(render_receipt(ReceiptSpec(seed=1)), 85)
    baseline = analyze_image(data, "x.jpg")
    names = [name for name, value in baseline["features"].items() if isinstance(value, (bool, int, float))][:6]
    model = {
        "model": "logistic", "features": names, "mean": [0.0] * len(names), "std": [1.0] * len(names),
        "coef": [0.0] * len(names), "intercept": 3.0,  # sigmoid(3) ≈ 0.95 → score 95
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model), encoding="utf-8")

    learned = analyze_image(data, "x.jpg", config=AnalysisConfig(scoring_model=str(path)))
    assert learned["score_model"] == "logistic:m.json" and learned["risk_score"] == 95
    assert learned["decision"] == "REVISAR" and learned["score_details"]["probability"] == 0.9526

    both = analyze_image(data, "x.jpg", config=AnalysisConfig(scoring_model=str(path), scoring_mode="max"))
    assert both["score_model"].startswith("max(") and both["risk_score"] == 95
    assert both["score_details"]["additive_score"] == baseline["risk_score"]

    missing = analyze_image(data, "x.jpg", config=AnalysisConfig(scoring_model=str(tmp_path / "nao_existe.json")))
    assert missing["score_model"] == "additive"
    assert isinstance(build_scorer(AnalysisConfig()), AdditiveScorer)
    assert isinstance(build_scorer(AnalysisConfig(scoring_model=str(path), scoring_mode="max")), MaxScorer)
