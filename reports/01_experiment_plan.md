# Experiment Plan

## Baseline

Train `configs/base_pretrain.yaml` with masked spectral reconstruction + small Lab auxiliary loss + smoothness prior.

## Objective ablations

- `objective_mlm_only`: reconstruction only
- `objective_deltae_aux`: stronger Lab auxiliary head
- compare validation reconstruction, DeltaE, retrieval

## Scaling ablations

- `scaling_tiny`: smaller transformer
- `base_multitask`: baseline transformer
- `scaling_small`: larger transformer
- plot validation loss and DeltaE vs trainable parameter count

## Data mixture ablations

Regenerate train/val/test splits into `data/mixtures/<mixture_name>` and point configs at those splits.

Recommended mixtures:

1. balanced
2. high_chroma_heavy
3. neutral_heavy
4. absorption_heavy
5. smooth_only

## Decision rules

- If reconstruction improves but DeltaE worsens, the objective is not perceptual enough.
- If scaling improves train loss but not val/test metrics, data diversity is likely the bottleneck.
- If retrieval improves but DeltaE worsens, embeddings may encode coarse hue but not perceptual distance.
