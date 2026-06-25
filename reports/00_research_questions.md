# Research Questions

## Primary question

Can masked spectral pretraining learn reusable color representations that transfer to perceptual color tasks?

## Sub-questions

1. Does auxiliary auxiliary target supervision improve downstream downstream metric metrics?
2. Does a smoothness prior help spectral reconstruction without hurting color retrieval?
3. How do model size and validation loss scale across tiny/small models?
4. Which synthetic data families create the largest generalization gap?
5. Do lower reconstruction losses correspond to lower perceptual color error?

## Hypotheses

- H1: masked modeling will reduce reconstruction error but may not automatically optimize perceptual downstream metric.
- H2: A small auxiliary target auxiliary loss should improve downstream metric-style evals.
- H3: Overweighting high-chroma synthetic spectra may improve color-name retrieval but hurt neutral spectra.
- H4: Larger models should reduce validation loss, but gains may saturate quickly under synthetic-data limits.
