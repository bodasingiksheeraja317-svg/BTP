"""
src/models/domain_generalization/coral.py
CORAL (Correlation Alignment) and MMD (Maximum Mean Discrepancy) DG methods.

Spec Section 12:
  Both attach at shared bottleneck z (256-d).
  CORAL: second-order covariance alignment across source domains, computed in-batch.
  MMD: maximum mean discrepancy alignment (shared file, alternate loss fn).
  Loss only — z shape is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def coral_loss(source_features: list[torch.Tensor]) -> torch.Tensor:
    """CORAL loss: minimize difference in covariance matrices across source domains.

    Args:
        source_features: List of feature tensors, one per source domain batch.
                         Each tensor shape: [N_i, 256].

    Returns:
        Scalar CORAL loss.
    """
    if len(source_features) < 2:
        dev = source_features[0].device if len(source_features) > 0 else torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=True)

    covs = [_compute_covariance(f) for f in source_features]

    dev = source_features[0].device
    loss = torch.tensor(0.0, device=dev)
    count = 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            diff = covs[i] - covs[j]
            loss = loss + (diff * diff).sum() / (4.0 * covs[i].shape[0] ** 2)
            count += 1

    return loss / max(count, 1)


def mmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    kernel: str = "rbf",
    gamma: float = 1.0,
) -> torch.Tensor:
    """MMD loss between source and target feature distributions.

    Args:
        source_features: [N_s, D]
        target_features: [N_t, D]
        kernel: 'rbf' (Gaussian RBF) or 'linear'.
        gamma: RBF kernel bandwidth parameter.

    Returns:
        Scalar MMD loss.
    """
    if kernel == "rbf":
        ss = _rbf_kernel(source_features, source_features, gamma)
        tt = _rbf_kernel(target_features, target_features, gamma)
        st = _rbf_kernel(source_features, target_features, gamma)
    elif kernel == "linear":
        ss = source_features @ source_features.T
        tt = target_features @ target_features.T
        st = source_features @ target_features.T
    else:
        raise ValueError(f"Unknown MMD kernel: {kernel}")

    return ss.mean() + tt.mean() - 2 * st.mean()


def _compute_covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute sample covariance matrix of x [N, D]."""
    n = x.shape[0]
    x_centered = x - x.mean(dim=0, keepdim=True)
    return x_centered.T @ x_centered / (n - 1)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Gaussian RBF kernel matrix."""
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True)
    xy = x @ y.T
    dist_sq = xx + yy.T - 2 * xy
    return torch.exp(-gamma * dist_sq)
