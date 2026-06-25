# FoundationPretrainLab Research Roadmap

## Phase 1: End-to-End Pretraining Loop

- [x] Synthetic train/validation/test splits
- [x] Masked-modeling baseline
- [x] Reconstruction evaluation
- [x] Smoke-run report

## Phase 2: Training Stability

- [ ] Normalize targets
- [ ] Tune loss weights
- [ ] Add learning-rate schedule experiments
- [ ] Add gradient clipping ablation
- [ ] Add longer CPU/GPU training run

## Phase 3: Objective Ablations

- [ ] Reconstruction-only pretraining
- [ ] Auxiliary-prediction baseline
- [ ] Multitask pretraining objective
- [ ] Contrastive objective

## Phase 4: Scaling Studies

- [ ] Model size sweep
- [ ] Dataset size sweep
- [ ] Masking-ratio sweep
- [ ] Data mixture sweep
- [ ] Loss vs samples curve

## Phase 5: Failure Analysis

- [ ] Error by feature region
- [ ] Error by sample type
- [ ] Reconstruction error vs downstream metric mismatch
- [ ] Out-of-distribution failure cases

## Phase 6: Public Research Report

- [ ] Initial technical report
- [ ] Scaling report
- [ ] Objective ablation report
- [ ] Failure analysis report
