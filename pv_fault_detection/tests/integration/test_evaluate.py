"""
tests/integration/test_evaluate.py
Integration tests: verify metrics.py and statistical_tests.py produce all required Section 14 fields.
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    FAULT_CLASSES,
    compute_all_metrics,
    compute_generalization_gap,
    compute_lodo_summary,
)
from src.evaluation.statistical_tests import (
    bootstrap_ci,
    wilcoxon_signed_rank,
)

pytestmark = pytest.mark.integration


def test_compute_all_metrics_schema():
    """Verify compute_all_metrics outputs every metric field required by Section 14.1."""
    N, C = 100, 7
    rng = np.random.default_rng(42)

    # Realistic binary ground truth and predictions
    y_true = rng.integers(0, 2, size=(N, C))
    y_logits = rng.normal(0, 2.0, size=(N, C))

    metrics = compute_all_metrics(y_true, y_logits, threshold=0.5)

    # Section 14.1 core required fields
    required_keys = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "accuracy",
        "balanced_accuracy",   # Added: spec Section 14.1 requires this
        "per_class",
        "per_class_recall",
        "hotspot_recall",
        "open_short_recall",
        "confusion_matrix",
        "false_positive_rate",
        "ece",
        "roc_auc_macro",
        "pr_auc_macro",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric field: {key}"

    # Verify per-class structure for all 7 canonical classes
    assert len(metrics["per_class"]) == 7
    for cls_name in FAULT_CLASSES:
        assert cls_name in metrics["per_class"]
        assert "precision" in metrics["per_class"][cls_name]
        assert "recall" in metrics["per_class"][cls_name]
        assert "f1" in metrics["per_class"][cls_name]


def test_cross_domain_metrics():
    """Verify generalization gap and LODO summary aggregation."""
    gap_metrics = compute_generalization_gap(source_macro_f1=0.85, unseen_macro_f1=0.70)
    assert gap_metrics["generalization_gap"] == pytest.approx(0.15, abs=1e-4)
    assert gap_metrics["relative_degradation"] == pytest.approx(0.15 / 0.85, abs=1e-4)

    # LODO summary across 3 rotations
    rotations = [
        {"held_out_domain": "d1", "macro_f1": 0.72, "source_macro_f1": 0.85},
        {"held_out_domain": "d2", "macro_f1": 0.68, "source_macro_f1": 0.84},
        {"held_out_domain": "d3", "macro_f1": 0.76, "source_macro_f1": 0.86},
    ]
    summary = compute_lodo_summary(rotations)
    assert summary["worst_case_domain_macro_f1"] == pytest.approx(0.68, abs=1e-4)
    assert summary["best_case_domain_macro_f1"] == pytest.approx(0.76, abs=1e-4)
    assert summary["mean_lodo_macro_f1"] == pytest.approx(0.72, abs=1e-4)
    assert summary["mean_source_macro_f1"] == pytest.approx(0.85, abs=1e-4)
    assert summary["mean_generalization_gap"] == pytest.approx(0.85 - 0.72, abs=1e-4)


def test_statistical_tests():
    """Verify bootstrap CI and Wilcoxon signed-rank test with effect size."""
    values = [0.82, 0.84, 0.83, 0.85, 0.81, 0.86]
    point, ci_low, ci_high = bootstrap_ci(values, n_resamples=500, confidence_level=0.95, seed=42)

    assert ci_low <= point <= ci_high
    assert 0.80 <= ci_low <= 0.84
    assert 0.83 <= ci_high <= 0.88

    # Paired Wilcoxon comparison: Config A (multimodal) vs Config B (single-modal)
    model_a_f1s = [0.86, 0.88, 0.85, 0.87, 0.89]
    model_b_f1s = [0.78, 0.80, 0.79, 0.81, 0.77]
    result = wilcoxon_signed_rank(model_a_f1s, model_b_f1s)

    assert "statistic" in result
    assert "p_value" in result
    assert "effect_size" in result
    assert result["effect_size"] > 0  # Model A consistently better than B
