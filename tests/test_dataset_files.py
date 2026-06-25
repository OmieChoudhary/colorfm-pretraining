from pathlib import Path

import pandas as pd


def test_train_val_test_files_exist_and_have_splits():
    for split, expected_rows in [("train", 800), ("val", 100), ("test", 100)]:
        path = Path("data") / f"{split}.csv"
        assert path.exists(), f"missing {path}"
        df = pd.read_csv(path)
        assert len(df) == expected_rows
        assert {"sample_id", "source", "color_name", "L", "a", "b"}.issubset(df.columns)
        spectral_cols = [c for c in df.columns if c.startswith("r_")]
        assert len(spectral_cols) == 41
        assert df[spectral_cols].min().min() >= 0.0
        assert df[spectral_cols].max().max() <= 1.0
