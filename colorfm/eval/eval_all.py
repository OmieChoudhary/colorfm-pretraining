from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colorfm.eval.eval_deltae import deltae_metrics
from colorfm.eval.eval_reconstruction import reconstruction_metrics
from colorfm.eval.eval_retrieval import retrieval_metrics
from colorfm.utils.experiment import save_json


def evaluate_all(checkpoint: str | Path, csv_path: str | Path) -> dict[str, object]:
    metrics: dict[str, object] = {}
    metrics.update(reconstruction_metrics(checkpoint, csv_path))
    metrics.update(deltae_metrics(checkpoint, csv_path))
    metrics.update(retrieval_metrics(checkpoint, csv_path))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    metrics = evaluate_all(args.checkpoint, args.csv)
    if args.out:
        save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
