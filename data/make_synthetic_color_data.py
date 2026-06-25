from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow running this script from the repo root without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from colorfm.utils.color_conversions import wavelengths_380_780_10, reflectance_to_lab

COLOR_NAMES = [
    "deep red", "orange", "warm yellow", "olive", "green", "cyan",
    "blue", "violet", "magenta", "brown", "gray", "near white"
]

DEFAULT_MIXTURE = {
    "smooth_basis": 0.40,
    "absorption_band": 0.25,
    "high_chroma": 0.20,
    "neutral_flat": 0.15,
}


def parse_mixture(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_MIXTURE)
    mixture: dict[str, float] = {}
    for item in text.split(","):
        name, weight = item.split(":")
        mixture[name.strip()] = float(weight)
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("Mixture weights must sum to a positive number")
    return {k: v / total for k, v in mixture.items()}


def choose_source(rng: np.random.Generator, mixture: dict[str, float]) -> str:
    names = list(mixture)
    probs = np.asarray([mixture[n] for n in names], dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _smooth_basis(rng: np.random.Generator, wl: np.ndarray) -> np.ndarray:
    base = rng.uniform(0.15, 0.85)
    slope = rng.normal(0.0, 0.12) * (wl - wl.mean()) / (wl.max() - wl.min())
    refl = np.full_like(wl, base, dtype=np.float32) + slope.astype(np.float32)
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(410, 740)
        width = rng.uniform(25, 95)
        amp = rng.uniform(-0.35, 0.35)
        refl += amp * np.exp(-0.5 * ((wl - center) / width) ** 2).astype(np.float32)
    return refl


def _absorption_band(rng: np.random.Generator, wl: np.ndarray) -> np.ndarray:
    refl = rng.uniform(0.55, 0.9) * np.ones_like(wl, dtype=np.float32)
    for _ in range(rng.integers(1, 4)):
        center = rng.uniform(430, 720)
        width = rng.uniform(10, 45)
        depth = rng.uniform(0.18, 0.55)
        refl -= depth * np.exp(-0.5 * ((wl - center) / width) ** 2).astype(np.float32)
    return refl


def _high_chroma(rng: np.random.Generator, wl: np.ndarray) -> np.ndarray:
    center = rng.uniform(420, 700)
    width = rng.uniform(35, 80)
    refl = 0.08 + rng.uniform(0.55, 0.9) * np.exp(-0.5 * ((wl - center) / width) ** 2).astype(np.float32)
    # Add shoulder to create more realistic broad-band pigments.
    shoulder = center + rng.uniform(-70, 70)
    refl += rng.uniform(0.05, 0.25) * np.exp(-0.5 * ((wl - shoulder) / rng.uniform(60, 120)) ** 2).astype(np.float32)
    return refl


def _neutral_flat(rng: np.random.Generator, wl: np.ndarray) -> np.ndarray:
    base = rng.uniform(0.05, 0.95)
    refl = np.full_like(wl, base, dtype=np.float32)
    refl += rng.normal(0, 0.015, size=wl.shape).astype(np.float32)
    return refl


GENERATORS = {
    "smooth_basis": _smooth_basis,
    "absorption_band": _absorption_band,
    "high_chroma": _high_chroma,
    "neutral_flat": _neutral_flat,
}


def make_spectrum(rng: np.random.Generator, wavelengths: np.ndarray, source: str) -> np.ndarray:
    if source not in GENERATORS:
        raise ValueError(f"Unknown synthetic source '{source}'. Options: {sorted(GENERATORS)}")
    refl = GENERATORS[source](rng, wavelengths.astype(np.float32))
    refl += rng.normal(0, 0.01, size=wavelengths.shape).astype(np.float32)
    return np.clip(refl, 0.02, 0.98).astype(np.float32)


def label_color_name(lab: np.ndarray) -> str:
    L, a, b = lab
    if L > 88:
        return "near white"
    if L < 28 and abs(a) < 12 and abs(b) < 12:
        return "gray"
    if abs(a) < 8 and abs(b) < 8:
        return "gray"
    hue = np.degrees(np.arctan2(b, a)) % 360
    if hue < 25 or hue >= 335:
        return "deep red"
    if hue < 65:
        return "orange"
    if hue < 105:
        return "warm yellow"
    if hue < 150:
        return "olive"
    if hue < 190:
        return "green"
    if hue < 230:
        return "cyan"
    if hue < 280:
        return "blue"
    if hue < 315:
        return "violet"
    return "magenta"


def make_split(n: int, seed: int, split: str, mixture: dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wavelengths = wavelengths_380_780_10()
    rows = []
    for i in range(n):
        source = choose_source(rng, mixture)
        spectrum = make_spectrum(rng, wavelengths, source)
        lab = reflectance_to_lab(spectrum, wavelengths)
        row = {
            "sample_id": f"{split}_{i:05d}",
            "source": source,
            "color_name": label_color_name(lab),
            "L": float(lab[0]),
            "a": float(lab[1]),
            "b": float(lab[2]),
        }
        for wl, val in zip(wavelengths.astype(int), spectrum):
            row[f"r_{wl}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixture",
        type=str,
        default=None,
        help="Comma-separated source weights, e.g. smooth_basis:0.5,absorption_band:0.2,high_chroma:0.2,neutral_flat:0.1",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mixture = parse_mixture(args.mixture)

    splits = {
        "train": (args.n_train, args.seed),
        "val": (args.n_val, args.seed + 1),
        "test": (args.n_test, args.seed + 2),
    }
    for split, (n, seed) in splits.items():
        df = make_split(n, seed, split, mixture)
        path = args.out_dir / f"{split}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path} with shape {df.shape} source_counts={df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
