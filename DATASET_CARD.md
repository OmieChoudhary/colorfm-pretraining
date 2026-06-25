# Dataset Card: Synthetic ColorFM Spectra

## Purpose

The dataset is used to test the mechanics of domain-specific foundation-model pretraining on spectral reflectance curves.

## Data source

Synthetic only. No proprietary, customer, or PPG data is included.

## Schema

Each row contains:

- `sample_id`
- `source`: synthetic generation family
- `color_name`: heuristic label from approximate Lab hue/lightness
- `L`, `a`, `b`: approximate Lab values
- `r_380` ... `r_780`: reflectance values on a 10 nm grid

## Splits

- `data/train.csv`: 800 rows
- `data/val.csv`: 100 rows
- `data/test.csv`: 100 rows

## Synthetic families

- `smooth_basis`: smooth low-frequency spectra
- `absorption_band`: high-reflectance spectra with absorption notches
- `high_chroma`: pigment-like peaked spectra
- `neutral_flat`: low-chroma near-flat spectra

## Known limitations

- Synthetic spectra are not a substitute for real spectrophotometer measurements.
- Color matching functions and illuminant curves are approximate.
- Generated color names are heuristic and should not be treated as ground truth.
- The data is useful for portfolio-scale pretraining research mechanics, not production color science.
