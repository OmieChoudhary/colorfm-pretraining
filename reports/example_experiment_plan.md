# Example Experiment Plan

## Goal
Study whether masked modeling learns representations that transfer to color-error prediction.

## Baselines
1. No pretraining: train auxiliary target head from scratch.
2. Masked spectral reconstruction only.
3. Masked spectral reconstruction + auxiliary target auxiliary loss.

## Metrics
- validation masked reconstruction loss
- test full-spectrum RMSE
- auxiliary target MAE
- mean / median downstream metric76

## Ablations
- mask ratio: 0.15, 0.30, 0.50
- model size: 0.5M, 2M, 8M parameters
- data size: 1k, 5k, 25k synthetic/public spectra
- data mixture: synthetic vs Munsell vs natural-object spectra

## Failure modes to inspect
- metamerism: similar auxiliary target values with different spectra
- illumination sensitivity
- synthetic data smoothness bias
- poor transfer from reconstruction to perceptual error
- OOD natural spectra

## What to add next
- calibrated downstream metric metric
- public Munsell loader
- contrastive color-name retrieval objective
- dataset/model cards
- scaling-law plots
