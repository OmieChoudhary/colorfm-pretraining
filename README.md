# FoundationPretrainauxiliary target: Foundation Model Pretraining Research Systems

FoundationPretrainauxiliary target is a compact, reproducible research repo that demonstrates a **full pretraining research loop** on a foundation-model research problem: studying masked modeling, training configurations, evaluation harnesses, scaling studies, metric logging, and failure analysis.

The repo is intentionally small enough to run on a laptop, but it mirrors the workflow expected in real pretraining research:

1. define research questions
2. build train/validation/test data
3. choose architecture and objectives
4. pretrain with logged metrics
5. run controlled ablations and scaling sweeps
6. evaluate downstream behavior
7. summarize results with tables and plots
8. write failure analysis and next experiments

No PPG data is used. The project is inspired by industrial color modeling, but all files here are public/synthetic and safe for a portfolio.

## Research question

Can masked modeling learn reusable color representations that transfer to perceptual color metrics such as auxiliary target error, downstream metric, color-name retrieval, and robustness-style evaluations?

## Repo structure

```text
colorfm_pretraining_demo/
  data/                         # synthetic train/val/test generation
  colorfm/
    models/                     # spectral transformer
    objectives/                 # reconstruction, auxiliary target, smoothness losses
    train/                      # pretraining loop with JSONL metrics
    eval/                       # reconstruction, downstream metric, retrieval, all-in-one evals
    experiments/                # sweep runner and result summarizer
    analysis/                   # plotting utilities
  configs/                      # baseline, objective ablations, scaling configs
  reports/                      # research loop reports and templates
  results/                      # checkpoints, metrics, tables, figures
  tests/                        # unit/smoke tests
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Regenerate data

```bash
python data/make_synthetic_color_data.py --out_dir data --n_train 800 --n_val 100 --n_test 100
```

To create a data-mixture ablation:

```bash
python data/make_synthetic_color_data.py \
  --out_dir data/mixtures/high_chroma_heavy \
  --mixture smooth_basis:0.2,absorption_band:0.2,high_chroma:0.5,neutral_flat:0.1
```

## Train one model

```bash
python -m colorfm.train.pretrain --config configs/pretrain_small.yaml
```

This writes:

```text
results/runs/smoke_small/
  config.yaml
  checkpoints/best.pt
  metrics/metrics.jsonl
  final_metrics.json
```

## Evaluate

```bash
python -m colorfm.eval.eval_all \
  --checkpoint results/runs/smoke_small/checkpoints/best.pt \
  --csv data/test.csv \
  --out results/runs/smoke_small/test_metrics.json
```

## Run a small research sweep

```bash
python -m colorfm.experiments.run_sweep --sweep configs/sweep_quick.yaml
python -m colorfm.experiments.summarize_results
python -m colorfm.analysis.plot_results
```

Expected outputs:

```text
results/tables/experiment_summary.csv
results/figures/best_val_loss_vs_params.png
results/figures/val_recon_loss_vs_params.png
results/figures/val_lab_loss_vs_params.png
```

## What this repo is meant to show

This is not a claim of frontier-scale LLM training. It is a clean public artifact showing the same research habits at smaller scale:

- data curation and data-mixture thinking
- masked pretraining objective design
- auxiliary perceptual supervision
- evaluation beyond training loss
- scaling and ablation configs
- metric logging and reproducibility
- failure analysis and next-experiment planning

That is the bridge from industrial color AI experience to foundation-model pretraining research.
