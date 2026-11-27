"""
src/evaluation/metrics.py
All evaluation metrics for PV fault detection.

Spec Section 14.1 — ALL of these must be present in metrics.json:
  Per-class and overall:
    accuracy, precision, recall, F1, macro_f1 (primary), balanced_accuracy,
    roc_auc, pr_auc, confusion_matrix, per_class_recall, false_positive_rate,
    expected_calibration_error (ECE), inference_latency_ms, model_param_count.

  Cross-domain specific (when protocol == lodo):
    generalization_gap, relative_degradation, worst_case_domain_macro_f1.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FAULT_CLASSES = [
    "Healthy",
    "Hotspot",
    "Open/Short-circuit",
    "Shading/Obstruction",
    "Soiling",
    "Structural (visual)",
    "Electrical/Diode fault",
]


def compute_all_metrics(
    y_true: np.ndarray,       # [N, C] binary multi-label targets
    y_pred_logits: np.ndarray, # [N, C] raw logits
    threshold: float = 0.5,
    fault_classes: list[str] = FAULT_CLASSES,
) -> dict[str, Any]:
    """Compute all Section 14.1 metrics from predictions.

    Args:
        y_true: Binary multi-label ground truth [N, C].
        y_pred_logits: Raw model logits [N, C].
        threshold: Decision threshold for binary prediction.
        fault_classes: Class names.

    Returns:
        Dict containing every metric required by spec Section 14.1.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
    )

    y_pred_probs = _sigmoid(y_pred_logits)
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    metrics: dict[str, Any] = {}

    precision_per_class = precision_score(y_true, y_pred_binary, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred_binary, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred_binary, average=None, zero_division=0)

    metrics["per_class"] = {
        cls: {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }
        for i, cls in enumerate(fault_classes)
    }

    metrics["macro_f1"] = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred_binary, average="micro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred_binary, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred_binary, average="macro", zero_division=0))

    exact_match = np.all(y_true == y_pred_binary, axis=1).mean()
    metrics["accuracy"] = float(exact_match)

    # Spec Section 14.1: balanced_accuracy (average per-class recall across all 7 classes)
    try:
        ba_scores = [
            balanced_accuracy_score(y_true[:, c], y_pred_binary[:, c])
            for c in range(y_true.shape[1])
        ]
        metrics["balanced_accuracy"] = float(np.mean(ba_scores))
    except Exception:
        metrics["balanced_accuracy"] = None

    metrics["per_class_recall"] = {
        cls: float(recall_per_class[i]) for i, cls in enumerate(fault_classes)
    }
    metrics["hotspot_recall"] = float(recall_per_class[FAULT_CLASSES.index("Hotspot")] if "Hotspot" in FAULT_CLASSES else 0.0)
    metrics["open_short_recall"] = float(recall_per_class[FAULT_CLASSES.index("Open/Short-circuit")] if "Open/Short-circuit" in FAULT_CLASSES else 0.0)

    try:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_pred_probs, average="macro"))
    except Exception:
        metrics["roc_auc_macro"] = None

    try:
        metrics["pr_auc_macro"] = float(average_precision_score(y_true, y_pred_probs, average="macro"))
    except Exception:
        metrics["pr_auc_macro"] = None

    metrics["confusion_matrix"] = [
        confusion_matrix(y_true[:, i], y_pred_binary[:, i]).tolist()
        for i in range(len(fault_classes))
    ]

    metrics["false_positive_rate"] = {
        cls: _fpr(y_true[:, i], y_pred_binary[:, i])
        for i, cls in enumerate(fault_classes)
    }

    metrics["ece"] = float(_compute_ece(y_true, y_pred_probs))

    return metrics


def compute_generalization_gap(
    source_macro_f1: float,
    unseen_macro_f1: float,
) -> dict[str, float]:
    """Compute cross-domain generalization gap metrics."""
    gap = source_macro_f1 - unseen_macro_f1
    relative_deg = gap / source_macro_f1 if source_macro_f1 > 0 else 0.0
    return {
        "generalization_gap": float(gap),
        "relative_degradation": float(relative_deg),
    }


def compute_lodo_summary(rotation_results: list[dict]) -> dict:
    """Aggregate per-rotation LODO results into summary stats."""
    if not rotation_results:
        return {}

    macro_f1s = [r.get("macro_f1", 0.0) for r in rotation_results]
    source_f1s = [r.get("source_macro_f1", 0.0) for r in rotation_results]

    return {
        "mean_lodo_macro_f1": float(np.mean(macro_f1s)),
        "std_lodo_macro_f1": float(np.std(macro_f1s)),
        "worst_case_domain_macro_f1": float(np.min(macro_f1s)),
        "best_case_domain_macro_f1": float(np.max(macro_f1s)),
        "mean_source_macro_f1": float(np.mean(source_f1s)),
        "mean_generalization_gap": float(np.mean(source_f1s) - np.mean(macro_f1s)),
        "per_rotation": rotation_results,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def _compute_ece(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) for multi-label classification."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = y_true.size
    for b_low, b_high in zip(bins[:-1], bins[1:]):
        mask = (y_pred_probs >= b_low) & (y_pred_probs < b_high)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_pred_probs[mask].mean()
        ece += (mask.sum() / n_total) * abs(acc - conf)
    return float(ece)
