from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SpectralColumns:
    wavelength_columns: list[str]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SpectralColumns":
        cols = sorted([c for c in df.columns if c.startswith("r_")], key=lambda x: int(x.split("_")[1]))
        if not cols:
            raise ValueError("No reflectance columns found. Expected columns like r_380, r_390, ...")
        return cls(cols)


class SpectralDataset(Dataset):
    """CSV-backed spectral reflectance dataset.

    Required columns:
      sample_id, source, color_name, L, a, b, r_380 ... r_780
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.columns = SpectralColumns.from_dataframe(self.df)
        required = {"sample_id", "source", "color_name", "L", "a", "b"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        self.spectra = self.df[self.columns.wavelength_columns].to_numpy(dtype=np.float32)
        self.lab = self.df[["L", "a", "b"]].to_numpy(dtype=np.float32)
        self.sample_ids = self.df["sample_id"].astype(str).tolist()
        self.sources = self.df["source"].astype(str).tolist()
        self.color_names = self.df["color_name"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "spectrum": torch.tensor(self.spectra[idx], dtype=torch.float32),
            "lab": torch.tensor(self.lab[idx], dtype=torch.float32),
            "sample_id": self.sample_ids[idx],
            "source": self.sources[idx],
            "color_name": self.color_names[idx],
        }


def make_mask(batch_size: int, seq_len: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    return torch.rand(batch_size, seq_len, device=device) < mask_ratio
