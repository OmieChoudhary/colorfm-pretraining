# Example Experiment Plan

## Goal
Study whether masked spectral modeling learns representations that transfer to color-error prediction.

## Baselines
1. No pretraining: train Lab head from scratch.
2. Masked spectral reconstruction only.
3. Masked spectral reconstruction + Lab auxiliary loss.

## Metrics
- validation masked reconstruction loss
- test full-spectrum RMSE
- Lab MAE
- mean / median DeltaE76

## Ablations
- mask ratio: 0.15, 0.30, 0.50
- model size: 0.5M, 2M, 8M parameters
- data size: 1k, 5k, 25k synthetic/public spectra
- data mixture: synthetic vs Munsell vs natural-object spectra

## Failure modes to inspect
- metamerism: similar Lab values with different spectra
- illumination sensitivity
- synthetic data smoothness bias
- poor transfer from reconstruction to perceptual error
- OOD natural spectra

## What to add next
- CIEDE2000 metric
- public Munsell loader
- contrastive color-name retrieval objective
- dataset/model cards
- scaling-law plots
