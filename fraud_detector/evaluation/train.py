"""Treino da regressão logística que combina as features dos sinais.

Entrada: o JSONL de resultados do harness (features + label por imagem) ou uma
lista de dicionários equivalente. Saída: modelo JSON legível, com médias e
desvios para padronização, coeficientes por feature e métricas de holdout.

Sem scikit-learn: gradiente descendente com regularização L2 e pesos de classe
balanceados, determinístico para o mesmo seed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..scoring import EXCLUDED_FEATURES, feature_value
from .metrics import best_threshold, pr_auc, roc_auc


def collect_feature_names(rows: Sequence[dict[str, Any]], max_categories: int = 8) -> list[str]:
    """Features numéricas/booleanas pelo nome; categóricas de baixa cardinalidade viram 'nome=valor'."""
    numeric: set[str] = set()
    categorical: dict[str, set[str]] = {}
    for row in rows:
        for name, value in row["features"].items():
            if name in EXCLUDED_FEATURES:
                continue
            if isinstance(value, str):
                categorical.setdefault(name, set()).add(value)
            elif isinstance(value, (bool, int, float)):
                numeric.add(name)
    names = sorted(numeric)
    for name, values in sorted(categorical.items()):
        if 1 < len(values) <= max_categories:
            names.extend(f"{name}={value}" for value in sorted(values))
    return names


def design_matrix(rows: Sequence[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.full((len(rows), len(names)), np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = feature_value(row["features"], name)
            if value is not None:
                matrix[i, j] = value
    return matrix


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 3000, learning_rate: float = 0.1,
                 balanced: bool = True) -> tuple[np.ndarray, float]:
    n, d = x.shape
    weights = np.zeros(d)
    bias = 0.0
    if balanced:
        positives = max(1, int(y.sum()))
        negatives = max(1, n - positives)
        sample_weight = np.where(y == 1, n / (2 * positives), n / (2 * negatives))
    else:
        sample_weight = np.ones(n)
    for _ in range(iterations):
        logits = x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = (probabilities - y) * sample_weight
        grad_w = x.T @ error / n + l2 * weights / n
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return weights, bias


def predict_probability(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ weights + bias, -30, 30)))


def stratified_split(labels: np.ndarray, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in (0, 1):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        cut = int(round(len(indices) * holdout))
        test_idx.extend(indices[:cut].tolist())
        train_idx.extend(indices[cut:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def train_model(rows: Sequence[dict[str, Any]], holdout: float = 0.25, seed: int = 0, l2: float = 1.0,
                iterations: int = 3000, cost_fn: float = 5.0, notes: str = "") -> dict[str, Any]:
    if len(rows) < 8:
        raise ValueError("são necessárias pelo menos 8 imagens rotuladas")
    names = collect_feature_names(rows)
    labels = np.array([int(row["label"]) for row in rows], dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("a base precisa ter imagens íntegras e forjadas")
    raw = design_matrix(rows, names)

    # Padronização calculada em toda a base (média imputa valores ausentes).
    mean = np.nanmean(raw, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    filled = np.where(np.isnan(raw), mean, raw)
    std = filled.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    x = (filled - mean) / std

    train_idx, test_idx = stratified_split(labels, holdout, seed)
    metrics: dict[str, Any] = {}
    if len(test_idx) >= 4 and labels[test_idx].sum() > 0 and labels[test_idx].sum() < len(test_idx):
        w, b = fit_logistic(x[train_idx], labels[train_idx], l2=l2, iterations=iterations)
        probabilities = predict_probability(x[test_idx], w, b)
        scores = (100 * probabilities).round().tolist()
        y_test = labels[test_idx].astype(int).tolist()
        metrics["holdout"] = {"n": int(len(test_idx)), "pr_auc": pr_auc(y_test, scores), "roc_auc": roc_auc(y_test, scores),
                              "best_threshold_cost_fn5": best_threshold(y_test, scores, 1.0, cost_fn)}

    weights, bias = fit_logistic(x, labels, l2=l2, iterations=iterations)
    probabilities = predict_probability(x, weights, bias)
    scores = (100 * probabilities).round().tolist()
    y_all = labels.astype(int).tolist()
    metrics["train"] = {"n": int(len(rows)), "pr_auc": pr_auc(y_all, scores), "roc_auc": roc_auc(y_all, scores),
                        "best_threshold_cost_fn5": best_threshold(y_all, scores, 1.0, cost_fn)}

    ranked = sorted(zip(names, weights.tolist()), key=lambda item: -abs(item[1]))
    return {
        "model": "logistic",
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(rows)),
        "positives": int(labels.sum()),
        "features": names,
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "coef": [round(float(v), 6) for v in weights],
        "intercept": round(float(bias), 6),
        "l2": l2,
        "metrics": metrics,
        "suggested_review_threshold": metrics["train"]["best_threshold_cost_fn5"]["threshold"],
        "top_features": [{"feature": name, "coef": round(coef, 4)} for name, coef in ranked[:15]],
        "notes": notes,
    }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_model(model: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
