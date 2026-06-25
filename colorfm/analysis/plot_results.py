from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    if metric not in df.columns or "model_params" not in df.columns:
        return
    plot_df = df.dropna(subset=[metric, "model_params"]).sort_values("model_params")
    if plot_df.empty:
        return
    plt.figure()
    plt.plot(plot_df["model_params"], plot_df[metric], marker="o")
    for _, row in plot_df.iterrows():
        plt.annotate(str(row.get("run_name", "run")), (row["model_params"], row[metric]), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Trainable parameters")
    plt.ylabel(metric)
    plt.title(f"{metric} vs model size")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/tables/experiment_summary.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    plot_metric(df, "best_val_loss", args.out_dir / "best_val_loss_vs_params.png")
    plot_metric(df, "val_recon_loss", args.out_dir / "val_recon_loss_vs_params.png")
    plot_metric(df, "val_lab_loss", args.out_dir / "val_lab_loss_vs_params.png")
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
