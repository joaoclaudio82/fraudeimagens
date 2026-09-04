"""Harness de avaliação: roda o analisador sobre uma base rotulada e mede cada sinal.

Uso:
    python scripts/evaluate.py --dataset data/synth --markdown docs/avaliacao.md

A base é uma pasta com imagens e um labels.jsonl (ver evaluation/synth.py). O
harness mantém um HashStore em memória e registra cada imagem depois de
analisá-la, reproduzindo o fluxo de produção em que reenvios são comparados
com o histórico.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..analyzer import analyze_image
from ..config import AnalysisConfig
from ..hashing import HashStore
from .metrics import best_threshold, binary_metrics, pr_auc, roc_auc

WEAK_CODES = {"ocr_unavailable"}


@dataclass
class ItemResult:
    file: str
    label: int
    kind: str
    origin: str
    score: int
    decision: str
    codes: list[str]
    features: dict[str, Any]
    region: dict[str, int] | None = None
    localized: bool | None = None
    errors: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_labels(dataset_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / "labels.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def region_hit(findings: list[dict[str, Any]], region: dict[str, int] | None) -> bool | None:
    """A região rotulada foi apontada por algum indicador com caixa? (centro dentro ou IoU >= 0.2)"""
    if not region:
        return None
    rx0, ry0 = region["x"], region["y"]
    rx1, ry1 = rx0 + region["w"], ry0 + region["h"]
    for item in findings:
        box = item.get("region")
        if not box:
            continue
        cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            return True
        ix = max(0, min(rx1, box["x"] + box["w"]) - max(rx0, box["x"]))
        iy = max(0, min(ry1, box["y"] + box["h"]) - max(ry0, box["y"]))
        inter = ix * iy
        union = region["w"] * region["h"] + box["w"] * box["h"] - inter
        if union and inter / union >= 0.2:
            return True
    return False


def run_dataset(dataset_dir: str | Path, config: AnalysisConfig | None = None, use_hash_store: bool = True,
                progress: Callable[[int, int, ItemResult], None] | None = None,
                analyze: Callable[..., dict[str, Any]] = analyze_image) -> list[ItemResult]:
    import time

    config = config or AnalysisConfig()
    folder = Path(dataset_dir)
    labels = load_labels(folder)
    store = HashStore() if use_hash_store else None
    results: list[ItemResult] = []
    for index, item in enumerate(labels):
        data = (folder / item["file"]).read_bytes()
        started = time.perf_counter()
        report = analyze(data, item["file"], item.get("expected") or {}, store, config)
        elapsed = time.perf_counter() - started
        if store is not None:
            store.add(report["sha256"], report["perceptual_hash"], report["features"]["duplicates.phash"],
                      item["file"], reference=(item.get("expected") or {}).get("pedido"))
        result = ItemResult(
            file=item["file"], label=int(item["label"]), kind=item.get("kind", "?"), origin=item.get("origin", "?"),
            score=int(report["risk_score"]), decision=report["decision"],
            codes=[f["code"] for f in report["findings"] if f["code"] not in WEAK_CODES],
            features=report["features"], region=item.get("region"),
            localized=region_hit(report["findings"], item.get("region")),
            errors=report.get("signal_errors", {}), seconds=round(elapsed, 3))
        results.append(result)
        if progress:
            progress(index + 1, len(labels), result)
    return results


def build_report(results: list[ItemResult], config: AnalysisConfig | None = None) -> dict[str, Any]:
    config = config or AnalysisConfig()
    y_true = [r.label for r in results]
    scores = [r.score for r in results]
    review = [int(r.score >= config.review_threshold) for r in results]
    attention = [int(r.score >= config.attention_threshold) for r in results]

    all_codes = sorted({code for r in results for code in r.codes})
    per_signal = {}
    for code in all_codes:
        pred = [int(code in r.codes) for r in results]
        per_signal[code] = binary_metrics(y_true, pred)

    kinds = sorted({r.kind for r in results if r.label == 1})
    per_kind = {}
    for kind in kinds:
        subset = [r for r in results if r.kind == kind]
        per_kind[kind] = {
            "n": len(subset),
            "recall_review": round(sum(r.score >= config.review_threshold for r in subset) / len(subset), 4),
            "recall_attention": round(sum(r.score >= config.attention_threshold for r in subset) / len(subset), 4),
            "mean_score": round(sum(r.score for r in subset) / len(subset), 1),
            "localized": _rate([r.localized for r in subset if r.localized is not None]),
            "top_codes": _top_codes(subset),
        }

    origins = sorted({r.origin for r in results if r.label == 0})
    per_origin = {}
    for origin in origins:
        subset = [r for r in results if r.label == 0 and r.origin == origin]
        per_origin[origin] = {
            "n": len(subset),
            "fp_review": round(sum(r.score >= config.review_threshold for r in subset) / len(subset), 4),
            "fp_attention": round(sum(r.score >= config.attention_threshold for r in subset) / len(subset), 4),
            "mean_score": round(sum(r.score for r in subset) / len(subset), 1),
            "top_codes": _top_codes(subset),
        }

    return {
        "n": len(results),
        "positives": int(sum(y_true)),
        "negatives": int(len(results) - sum(y_true)),
        "thresholds": {"review": config.review_threshold, "attention": config.attention_threshold},
        "combined": {
            "pr_auc": pr_auc(y_true, scores),
            "roc_auc": roc_auc(y_true, scores),
            "at_review": binary_metrics(y_true, review),
            "at_attention": binary_metrics(y_true, attention),
            "best_threshold_cost_fn5": best_threshold(y_true, scores, cost_fp=1.0, cost_fn=5.0),
        },
        "per_signal": per_signal,
        "per_kind": per_kind,
        "per_origin": per_origin,
        "errors": sorted({f"{k}: {v}" for r in results for k, v in r.errors.items()}),
        "mean_seconds": round(sum(r.seconds for r in results) / max(1, len(results)), 3),
    }


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _top_codes(subset: list[ItemResult], limit: int = 4) -> list[str]:
    counts: dict[str, int] = {}
    for r in subset:
        for code in r.codes:
            counts[code] = counts.get(code, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:limit]
    return [f"{code} ({count}/{len(subset)})" for code, count in ranked]


def report_markdown(report: dict[str, Any], title: str = "Avaliação") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Itens: {report['n']} ({report['positives']} forjados, {report['negatives']} íntegros). "
                 f"Tempo médio por imagem: {report['mean_seconds']} s.")
    combined = report["combined"]
    lines += ["", "## Score combinado", "", "| Métrica | Valor |", "|---|---|",
              f"| PR-AUC | {combined['pr_auc']} |", f"| ROC-AUC | {combined['roc_auc']} |"]
    for name, key in (("REVISAR", "at_review"), ("ATENÇÃO", "at_attention")):
        m = combined[key]
        lines.append(f"| Limiar {name} ({report['thresholds'][key.split('_')[1]]}): precisão / recall / FPR | "
                     f"{m['precision']} / {m['recall']} / {m['fpr']} |")
    best = combined["best_threshold_cost_fn5"]
    lines.append(f"| Melhor limiar (FN custa 5× FP) | {best['threshold']:.0f} (precisão {best['precision']}, recall {best['recall']}) |")

    lines += ["", "## Por indicador", "", "| Indicador | Acionou | Precisão | Recall | FPR |", "|---|---|---|---|---|"]
    for code, m in sorted(report["per_signal"].items(), key=lambda item: -item[1]["f1"]):
        lines.append(f"| {code} | {m['flagged']} | {m['precision']} | {m['recall']} | {m['fpr']} |")

    lines += ["", "## Recall por tipo de forjaria", "", "| Tipo | n | REVISAR | ATENÇÃO | Score médio | Localizou região | Indicadores mais frequentes |",
              "|---|---|---|---|---|---|---|"]
    for kind, m in report["per_kind"].items():
        localized = "n/a" if m["localized"] is None else m["localized"]
        lines.append(f"| {kind} | {m['n']} | {m['recall_review']} | {m['recall_attention']} | {m['mean_score']} | {localized} | {', '.join(m['top_codes'])} |")

    lines += ["", "## Falsos positivos por origem da imagem íntegra", "", "| Origem | n | FP em REVISAR | FP em ATENÇÃO | Score médio | Indicadores mais frequentes |",
              "|---|---|---|---|---|---|"]
    for origin, m in report["per_origin"].items():
        lines.append(f"| {origin} | {m['n']} | {m['fp_review']} | {m['fp_attention']} | {m['mean_score']} | {', '.join(m['top_codes'])} |")

    if report["errors"]:
        lines += ["", "## Erros de sinal", ""] + [f"- {error}" for error in report["errors"]]
    return "\n".join(lines) + "\n"


def save_results(results: list[ItemResult], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[ItemResult]:
    with Path(path).open(encoding="utf-8") as handle:
        return [ItemResult(**json.loads(line)) for line in handle if line.strip()]
