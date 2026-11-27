# Experiment Configurations

This directory contains configuration files for the experiments described in Section 15 (Ablation Matrix) and Section 18 (Config System) of the specification.

## Base Configuration

- `base.yaml`: Canonical base configuration containing all default values across all pipeline components. Every experiment configuration merges on top of this file.

## Ablation Matrix Configurations (Section 15)

| Row | Config File | Description | CLI Overrides | Protocol |
|---|---|---|---|---|
| 1 | `baseline_rgb.yaml` | RGB-only baseline | None | random_split |
| 2 | `baseline_thermal.yaml` | Thermal-only baseline | None | random_split |
| 3 | `baseline_electrical.yaml` | Electrical-only baseline | None | random_split |
| 4 | `multimodal_rgb_thermal.yaml` | RGB + Thermal bimodal | None | random_split |
| 5 | `multimodal_full.yaml` | Full trimodal (RGB+Thermal+Electrical) | None | random_split |
| 6 | `fusion_late_vs_crossattn.yaml` | Late vs cross-attention fusion | `--override model.fusion_type=late` vs `cross_attention` | random_split |
| 7 | `synthetic_ablation.yaml` | Real vs Real+Synthetic ratio sweep | `--override synthetic.ratio=0.0,0.1,0.25,0.5,1.0,...` | random_split |
| 8 | `domain_generalization.yaml` | DG method comparison | `--override model.dg_method=none,coral,adversarial,domain_norm` | lodo |
| 9 | `lodo_final.yaml` | In-domain vs LODO cross-site claim | `--override evaluation.protocol=random_split` vs `lodo` | both |

## Running Experiments

```bash
# Run a single experiment
python -m src.experiments.run --config configs/baseline_thermal.yaml

# Run with overrides
python -m src.experiments.run --config configs/synthetic_ablation.yaml --override synthetic.ratio=0.25 seed=1

# Run full ablation matrix
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml
```
