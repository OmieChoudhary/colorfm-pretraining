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


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def retrieval_metrics(checkpoint: str | Path, csv_path: str | Path, batch_size: int = 64) -> dict[str, float | str]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    ds = SpectralDataset(csv_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = SpectralTransformer(seq_len=ckpt["seq_len"], **cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["spectrum"], mask=None)
            embeddings.append(out["embedding"].numpy())
    emb = _normalize(np.concatenate(embeddings))
    labels = np.asarray(ds.color_names)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    ranks = np.argsort(-sims, axis=1)

    def recall_at(k: int) -> float:
        top = ranks[:, :k]
        hits = [labels[i] in labels[top[i]] for i in range(len(labels))]
        return float(np.mean(hits))

    return {
        "checkpoint": str(checkpoint),
        "csv_path": str(csv_path),
        "color_name_recall_at_1": recall_at(1),
        "color_name_recall_at_5": recall_at(min(5, len(labels) - 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv or Path("data") / f"{args.split}.csv"
    metrics = retrieval_metrics(args.checkpoint, csv_path)
    if args.out:
        save_json(metrics, args.out)
    print(metrics)


if __name__ == "__main__":
    main()
