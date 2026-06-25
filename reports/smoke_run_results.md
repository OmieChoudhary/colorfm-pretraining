# Smoke Pretraining Run Results

## Goal

Validate that the ColorFM pretraining loop runs end-to-end on CPU using the small synthetic spectral dataset.

This smoke run is not intended to demonstrate strong model quality. It verifies that the repository supports a complete pretraining research loop:

1. load train/validation/test spectral data
2. initialize a masked spectral transformer
3. train a reconstruction + Lab prediction objective
4. save a checkpoint
5. run reconstruction evaluation
6. run DeltaE evaluation

## Configuration

- Config: `configs/pretrain_small.yaml`
- Device: CPU
- Dataset: synthetic spectral train/validation/test splits
- Objective: masked spectral reconstruction + Lab prediction
- Checkpoint: `results/runs/smoke_small/checkpoints/best.pt`

## Test Status

```text
9 passed

Reconstruction Evaluation
full_spectrum_rmse: 1.2265
full_spectrum_mae: 1.2109
p95_abs_error: 1.5270

DeltaE Evaluation
mean_delta_e_76: 102.8033
median_delta_e_76: 102.3855
mean_delta_e_2000: 71.9769
median_delta_e_2000: 72.2092
p95_delta_e_2000: 96.7534
lab_mae_L: 75.4809
lab_mae_a: 60.8650
lab_mae_b: 22.2079
