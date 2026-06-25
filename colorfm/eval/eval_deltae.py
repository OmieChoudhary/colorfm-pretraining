from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colorfm.data import SpectralDataset
from colorfm.models.spectral_transformer import SpectralTransformer
from colorfm.utils.color_conversions import delta_e_76, delta_e_2000
from colorfm.utils.experiment import save_json


def deltae_metrics(checkpoint: str | Path, csv_path: str | Path, batch_size: int = 64) -> dict[str, float | str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    ds = SpectralDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = SpectralTransformer(seq_len=ckpt["seq_len"], **cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_labs, true_labs = [], []
    with torch.no_grad():
        for batch in loader:
            spectrum = batch["spectrum"]
            out = model(spectrum, mask=None)
            pred_labs.append(out["lab"].numpy())
            true_labs.append(batch["lab"].numpy())
    pred = np.concatenate(pred_labs)
    true = np.concatenate(true_labs)
    de76 = delta_e_76(pred, true)
    de00 = delta_e_2000(pred, true)
    mae = np.abs(pred - true).mean(axis=0)
    return {
        "checkpoint": str(checkpoint),
        "csv_path": str(csv_path),
        "mean_delta_e_76": float(de76.mean()),
        "median_delta_e_76": float(np.median(de76)),
        "mean_delta_e_2000": float(de00.mean()),
        "median_delta_e_2000": float(np.median(de00)),
        "p95_delta_e_2000": float(np.percentile(de00, 95)),
        "lab_mae_L": float(mae[0]),
        "lab_mae_a": float(mae[1]),
        "lab_mae_b": float(mae[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv or Path("data") / f"{args.split}.csv"
    metrics = deltae_metrics(args.checkpoint, csv_path)
    if args.out:
        save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
