"""Gera uma base sintética rotulada de comprovantes íntegros e forjados.

    python scripts/generate_dataset.py --out data/synth --intact 60 --forged 60 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_detector.evaluation.synth import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="pasta de saída")
    parser.add_argument("--intact", type=int, default=30)
    parser.add_argument("--forged", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whatsapp-fraction", type=float, default=0.3,
                        help="fração das forjarias reenviadas por WhatsApp (redimensiona e recomprime)")
    args = parser.parse_args()

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total}", end="", flush=True)

    items = generate_dataset(args.out, args.intact, args.forged, args.seed, args.whatsapp_fraction, progress)
    print(f"\n{len(items)} imagens em {args.out} (labels.jsonl gerado)")


if __name__ == "__main__":
    main()
