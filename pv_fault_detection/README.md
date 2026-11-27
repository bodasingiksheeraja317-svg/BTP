# PV Fault Detection — Multimodal Cross-Site Research Codebase

A research codebase for **multimodal PV (photovoltaic) panel fault detection** combining RGB, thermal, and physics-derived electrical features, with rigorous cross-site generalization evaluation.

## Research Questions

1. **RQ1 (Fusion):** Does fusing RGB + thermal + electrical beat the best single-modality baseline, in-domain and under domain shift?
2. **RQ2 (Generalization gap):** How large is the accuracy/F1 gap between source-domain and unseen-site performance, and does multimodality reduce it?
3. **RQ3 (Synthetic data):** Does physics-based + generative synthetic data reduce the cross-site generalization gap?
4. **RQ4 (DG technique value):** Among CORAL / adversarial-DG / domain-specific normalization, which (if any) beats plain pooled-ERM on LODO evaluation?

## Quickstart

```bash
# 1. Environment
pip install -e .

# 2. Data acquisition
python -m src.data.download --config configs/base.yaml
# Check data/DATASET_STATUS.md for which datasets need manual placement
# Then re-run after placing manual datasets under data/raw/<dataset_id>/

# 3. Canonical indexing
python -m src.data.indexer --config configs/base.yaml

# 4. Splitting
python -m src.splitting.lodo_split --config configs/base.yaml

# 5. Synthetic generation (cached)
python -m src.synthetic.electrical_sim --config configs/base.yaml
python -m src.synthetic.image_diffusion --config configs/base.yaml

# 6. Single experiment
python -m src.experiments.run --config configs/baseline_thermal.yaml

# 7. Full ablation matrix
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml

# 8. Reports
python -m src.evaluation.statistical_tests --results_dir results/
python -m src.evaluation.report_tables --results_dir results/ --out reports/
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data & Schema](docs/data.md)
- [Datasets & Acquisition](docs/datasets.md)
- [Training](docs/training.md)
- [Evaluation & LODO](docs/evaluation.md)
- [Synthetic Data](docs/synthetic_data.md)
- [Domain Generalization](docs/domain_generalization.md)

## Specification

See `PROJECT_IMPLEMENTATION_SPEC.md` (at repository root, one level up) for the complete single source of truth.

## Status

See [`STATUS.md`](STATUS.md) for current build stage, what's done, what's blocked, and open assumptions.
