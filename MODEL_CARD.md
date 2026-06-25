# Model Card: SpectralTransformer

## Model

A small transformer encoder for masked spectral modeling.

Input:

- Reflectance spectrum over 380-780 nm, sampled every 10 nm.

Outputs:

- Reconstructed reflectance spectrum
- Predicted Lab vector
- Pooled embedding for retrieval/evaluation

## Objectives

- Masked spectral reconstruction
- Optional Lab auxiliary loss
- Optional spectral smoothness regularization

## Intended use

This model is for research-loop demonstration: ablations, scaling sweeps, and eval harnesses for domain-specific pretraining.

## Not intended use

- Production color matching
- Product formulation recommendations
- Scientific claims about real pigments without real validation data

## Evaluation metrics

- Full-spectrum RMSE and MAE
- DeltaE76 and CIEDE2000 between predicted and approximate target Lab
- Color-name retrieval Recall@K
- Validation loss and objective-specific losses

## Limitations

- Small model and small synthetic data
- Approximate color science utilities
- No real industrial measurements
