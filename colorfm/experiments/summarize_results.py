from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_last_jsonl(path: Path) -> dict:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/experiment_summary.csv"))
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.runs_dir.glob("*")):
        final_path = run_dir / "final_metrics.json"
        metrics_path = run_dir / "metrics" / "metrics.jsonl"
        config_path = run_dir / "config.yaml"
        if not final_path.exists() or not metrics_path.exists():
            continue
        final = json.loads(final_path.read_text())
        last = read_last_jsonl(metrics_path)
        row = {**final, **last, "config_path": str(config_path)}
        rows.append(row)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(df)}")
    if len(df):
        print(df[[c for c in ["run_name", "model_params", "best_val_loss", "val_recon_loss", "val_lab_loss"] if c in df.columns]])


if __name__ == "__main__":
    main()
