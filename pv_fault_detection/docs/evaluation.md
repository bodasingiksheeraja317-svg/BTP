# Evaluation

## Metrics (spec Section 14.1)

All metrics are computed by `src/evaluation/metrics.py` and written to
`results/<experiment_id>/metrics.json`.

| Metric | Notes |
|---|---|
| macro_f1 | PRIMARY metric (handles class imbalance) |
| per_class_recall | Especially tracked for Hotspot and Open/Short-circuit (safety-critical) |
| accuracy, precision, recall, F1 | Per-class and overall |
| balanced_accuracy | For imbalanced evaluation |
| roc_auc, pr_auc | Macro-averaged |
| confusion_matrix | Per class (one-vs-rest) |
| false_positive_rate | Per class |
| ece | Expected Calibration Error |
| inference_latency_ms | ms/sample on eval hardware |
| model_param_count | Parameter count |

## Cross-Domain Metrics (LODO only)

| Metric | Definition |
|---|---|
| generalization_gap | mean(source held-out macro-F1) - unseen-site macro-F1 |
| relative_degradation | generalization_gap / source_performance |
| worst_case_domain_macro_f1 | min(macro_F1 across all LODO rotations) |

## LODO Protocol

For each domain Di:
1. Train fresh on ALL other domains (train portion only)
2. Internal val carved from training domains only (Di never seen)
3. Evaluate ONCE on Di's test set
4. Aggregate mean ± std across all rotations

See `src/evaluation/lodo_runner.py`.

## Statistical Validation

- Minimum 5 seeds per configuration (config `n_seeds`)
- Bootstrap 1,000 resamples for 95% CI on every headline metric
- Wilcoxon signed-rank test (paired) for all two-configuration comparisons
- Effect size (rank-biserial correlation) alongside every p-value
