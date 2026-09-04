"""Avalia o analisador sobre uma base rotulada e escreve relatório JSON/Markdown.

    python scripts/evaluate.py --dataset data/synth --markdown docs/avaliacao.md --results data/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_detector.config import AnalysisConfig  # noqa: E402
from fraud_detector.evaluation.harness import build_report, report_markdown, run_dataset, save_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", help="JSON com overrides de AnalysisConfig")
    parser.add_argument("--json", help="arquivo de saída do relatório completo")
    parser.add_argument("--markdown", help="arquivo de saída do resumo em Markdown")
    parser.add_argument("--results", help="JSONL com score, decisão, indicadores e features por imagem (entrada do treino)")
    parser.add_argument("--title", default="Avaliação em base sintética")
    parser.add_argument("--no-hash-store", action="store_true", help="não compara com as imagens anteriores")
    args = parser.parse_args()

    config = AnalysisConfig()
    if args.config:
        config = AnalysisConfig.from_dict({**config.to_dict(), **json.loads(Path(args.config).read_text(encoding="utf-8"))})

    def progress(done: int, total: int, item) -> None:
        print(f"\r{done}/{total} {item.file[:40]:40s} score={item.score:3d} {item.decision:12s}", end="", flush=True)

    results = run_dataset(args.dataset, config, not args.no_hash_store, progress)
    print()
    report = build_report(results, config)
    markdown = report_markdown(report, args.title)
    print(markdown)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    if args.results:
        save_results(results, args.results)


if __name__ == "__main__":
    main()
