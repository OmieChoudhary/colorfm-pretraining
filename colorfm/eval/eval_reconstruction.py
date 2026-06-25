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
from colorfm.utils.experiment import save_json


def reconstruction_metrics(checkpoint: str | Path, csv_path: str | Path, batch_size: int = 64) -> dict[str, float | str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    ds = SpectralDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = SpectralTransformer(seq_len=ckpt["seq_len"], **cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sqerr, abserr = [], []
    with torch.no_grad():
        for batch in loader:
            spectrum = batch["spectrum"]
            out = model(spectrum, mask=None)
            err = (out["reconstruction"] - spectrum).numpy()
            sqerr.append(err**2)
            abserr.append(np.abs(err))
    sq = np.concatenate(sqerr)
    ae = np.concatenate(abserr)
    return {
        "checkpoint": str(checkpoint),
        "csv_path": str(csv_path),
        "full_spectrum_rmse": float(np.sqrt(sq.mean())),
        "full_spectrum_mae": float(ae.mean()),
        "p95_abs_error": float(np.percentile(ae, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv or Path("data") / f"{args.split}.csv"
    metrics = reconstruction_metrics(args.checkpoint, csv_path)
    if args.out:
        save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
