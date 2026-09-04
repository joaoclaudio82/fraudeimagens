"""Métricas de classificação em numpy puro, sem dependência de scikit-learn."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4), "flagged": tp + fp}


def pr_curve(y_true: Sequence[int], scores: Sequence[float]) -> list[tuple[float, float, float]]:
    """Pontos (recall, precision, limiar) varrendo os limiares do maior para o menor."""
    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positives = int(truth.sum())
    if positives == 0 or len(values) == 0:
        return []
    order = np.argsort(-values, kind="stable")
    truth = truth[order]
    values = values[order]
    points: list[tuple[float, float, float]] = []
    tp = fp = 0
    index = 0
    while index < len(values):
        threshold = values[index]
        while index < len(values) and values[index] == threshold:
            tp += int(truth[index] == 1)
            fp += int(truth[index] == 0)
            index += 1
        points.append((tp / positives, tp / (tp + fp), float(threshold)))
    return points


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision: soma de (ΔR) × P sobre os limiares (equivale à área sob a curva PR em degraus)."""
    points = pr_curve(y_true, scores)
    if not points:
        return 0.0
    area = 0.0
    previous_recall = 0.0
    for recall, precision, _ in points:
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return round(float(area), 4)


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Estatística de Mann-Whitney com tratamento de empates."""
    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positives = values[truth == 1]
    negatives = values[truth == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return round(float((greater + 0.5 * ties) / (len(positives) * len(negatives))), 4)


def best_threshold(y_true: Sequence[int], scores: Sequence[float], cost_fp: float = 1.0, cost_fn: float = 5.0
                   ) -> dict[str, Any]:
    """Limiar que minimiza custo_fp × FP + custo_fn × FN, útil para calibrar a fila de revisão."""
    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    candidates = sorted(set(values.tolist()) | {float(values.max()) + 1 if len(values) else 1.0})
    best: dict[str, Any] | None = None
    for threshold in candidates:
        pred = (values >= threshold).astype(int)
        metrics = binary_metrics(truth, pred)
        cost = cost_fp * metrics["fp"] + cost_fn * metrics["fn"]
        if best is None or cost < best["cost"]:
            best = {"threshold": float(threshold), "cost": float(cost), **metrics}
    return best or {"threshold": 0.0, "cost": 0.0}
