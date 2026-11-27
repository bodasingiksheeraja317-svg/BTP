"""
src/training/losses.py
Loss functions for PV fault detection training.

Spec Section 13.2:
  - multilabel_focal_loss: primary default
  - class_weighted_bce_loss: alternative
  - coral_loss, mmd_loss: from models/domain_generalization/coral.py
  - adversarial_domain_loss: cross-entropy on domain classifier output

Total loss assembly happens in trainer.py, not here.
This module only provides the building blocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def multilabel_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-label focal loss for class-imbalanced PV fault classification.

    Args:
        logits: [B, C] raw (pre-sigmoid) logits.
        targets: [B, C] binary target labels (multi-label).
        alpha: Focal loss alpha (balance factor).
        gamma: Focal loss gamma (focusing parameter; default 2.0 per spec ASSUMPTION).
        reduction: 'mean' or 'sum'.

    Returns:
        Scalar loss.
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p_t = targets * probs + (1 - targets) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

    loss = alpha_t * focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    return loss.sum()


def class_weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-weighted binary cross-entropy for multi-label classification."""
    if class_weights is not None:
        weight = class_weights.to(logits.device)
    else:
        weight = None
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="mean"
    )


def adversarial_domain_loss(
    domain_logits: torch.Tensor,
    domain_targets: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss on domain classifier output."""
    return F.cross_entropy(domain_logits, domain_targets)
