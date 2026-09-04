"""Treina a regressão logística do score a partir do JSONL produzido por scripts/evaluate.py.

    python scripts/train_scoring.py --results data/results.jsonl --out models/scoring_logistic.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_detector.evaluation.train import load_rows, save_model, train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, nargs="+", help="um ou mais JSONL de resultados (podem ser bases diferentes)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--cost-fn", type=float, default=5.0, help="custo de um falso negativo em relação a um falso positivo")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    rows = [row for path in args.results for row in load_rows(path)]
    model = train_model(rows, holdout=args.holdout, seed=args.seed, l2=args.l2, cost_fn=args.cost_fn, notes=args.notes)
    save_model(model, args.out)

    print(f"{model['n_train']} imagens ({model['positives']} forjadas), {len(model['features'])} features")
    for split, metrics in model["metrics"].items():
        best = metrics["best_threshold_cost_fn5"]
        print(f"{split:8s} PR-AUC={metrics['pr_auc']}  ROC-AUC={metrics['roc_auc']}  "
              f"melhor limiar={best['threshold']:.0f} (precisão {best['precision']}, recall {best['recall']})")
    print("features mais influentes:")
    for item in model["top_features"][:10]:
        print(f"  {item['coef']:+.3f}  {item['feature']}")
    print(f"modelo salvo em {args.out}; ative com AnalysisConfig(scoring_model='{args.out}')")


if __name__ == "__main__":
    main()
