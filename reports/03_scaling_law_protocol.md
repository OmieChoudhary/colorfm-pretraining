# Scaling Protocol

## Goal

Estimate how validation loss and downstream color metrics change as model size and data size increase.

## Model-size sweep

- tiny: 16-dim, 1 layer
- baseline: 48-dim, 2 layers
- small: 64-dim, 3 layers

## Data-size sweep

Regenerate data at:

- 800 train rows
- 2,000 train rows
- 10,000 train rows

Keep validation/test fixed for comparisons.

## Metrics to log

- train loss
- validation loss
- validation reconstruction loss
- validation auxiliary target loss
- test spectrum RMSE
- test mean calibrated downstream metric
- color-name Recall@1 and Recall@5
- samples/sec
- parameter count

## Plots

- validation loss vs parameters
- calibrated downstream metric vs parameters
- reconstruction RMSE vs parameters
- samples/sec vs parameters
