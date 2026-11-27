# Training

## Pipeline Stages (spec Section 13.1)

1. Raw data -> Synchronization -> Cleaning -> Calibration -> Quality control -> Labeling
2. Preprocessing -> Augmentation -> Synthetic generation/mixing
3. Domain-aware split (site-level, never sample-level; LODO rotation)
4. Modality-specific preprocessing -> Model training
5. Hyperparameter optimization (Optuna, budget-limited) -> Model selection
6. Evaluation

## Launching a Run

```bash
# Single experiment
python -m src.experiments.run --config configs/baseline_thermal.yaml

# With overrides
python -m src.experiments.run --config configs/synthetic_ablation.yaml --override synthetic.ratio=0.25 seed=1

# Full ablation matrix
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml
```

## Config Key Reference

See `configs/base.yaml` for all keys with defaults and comments.

Key config sections:
- `model.*`: architecture (backbone, fusion_type, dg_method, modalities)
- `training.*`: optimizer, LR, epochs, loss type, DG weights
- `synthetic.*`: ratio, SSIM threshold
- `evaluation.*`: protocol (random_split or lodo)
- `hardware.*`: device, mixed_precision
