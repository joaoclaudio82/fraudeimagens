"""Linha de comando: análise em lote, servidor HTTP e manutenção.

    python -m fraud_detector.cli analyze fotos/*.jpg --expected pedido=12345 --db data/imageguard.sqlite
    python -m fraud_detector.cli serve --port 8000
    python -m fraud_detector.cli purge --db data/imageguard.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import AnalysisConfig, analyze_image
from .storage import AnalysisStore

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def iter_images(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.exists():
            files.append(path)
    return files


def parse_expected(items: list[str] | None) -> dict[str, str]:
    expected: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if sep:
            expected[key.strip()] = value.strip()
    return expected


def load_config(path: str | None, model: str | None) -> AnalysisConfig:
    values: dict[str, Any] = {}
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    if model:
        values["scoring_model"] = model
    return AnalysisConfig.from_dict({**AnalysisConfig().to_dict(), **values})


def cmd_analyze(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.model)
    store = AnalysisStore(args.db) if args.db else None
    expected = parse_expected(args.expected)
    files = iter_images(args.paths)
    if not files:
        print("nenhuma imagem encontrada", file=sys.stderr)
        return 1
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    print(f"{'arquivo':40s} {'score':>5s} {'decisão':12s} indicadores")
    for path in files:
        report = analyze_image(path.read_bytes(), path.name, expected, store.hashes if store else None, config)
        report.pop("_images", None)
        report.pop("_ela_image", None)
        if store:
            report["analysis_id"] = store.save(report, args.reference or expected.get("pedido"))
        codes = ", ".join(f["code"] for f in report["findings"] if f["points"]) or "-"
        print(f"{path.name[:40]:40s} {report['risk_score']:5d} {report['decision']:12s} {codes}")
        if out:
            out.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
    if out:
        out.close()
    if store:
        store.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("fraud_detector.api:app", host=args.host, port=args.port, reload=False)
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    store = AnalysisStore(args.db)
    removed = store.purge_expired()
    print(f"{removed} análise(s) expirada(s) removida(s); restam {store.stats()['analyses']}")
    store.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    store = AnalysisStore(args.db)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud_detector", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analisa arquivos ou pastas")
    analyze.add_argument("paths", nargs="+")
    analyze.add_argument("--expected", action="append", metavar="CAMPO=VALOR")
    analyze.add_argument("--reference", help="identificador do pedido para o histórico")
    analyze.add_argument("--db", help="SQLite para histórico, hashes e fila de revisão")
    analyze.add_argument("--out", help="JSONL com os relatórios completos")
    analyze.add_argument("--config", help="JSON com overrides de AnalysisConfig")
    analyze.add_argument("--model", help="modelo JSON treinado para o score")
    analyze.set_defaults(func=cmd_analyze)

    serve = sub.add_parser("serve", help="sobe a API HTTP")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    purge = sub.add_parser("purge", help="remove análises além da retenção")
    purge.add_argument("--db", required=True)
    purge.set_defaults(func=cmd_purge)

    stats = sub.add_parser("stats", help="resumo do histórico e da fila")
    stats.add_argument("--db", required=True)
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
