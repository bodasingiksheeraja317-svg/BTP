"""
src/models/domain_generalization/domain_norm.py
Domain-specific batch normalization for the bottleneck MLP.

Spec Section 12:
  Per-domain BatchNorm statistics in the bottleneck MLP.
  Swaps LayerNorm for domain-indexed BatchNorm — purely architectural, no extra loss.
  Each domain gets its own running mean/var in its BatchNorm layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DomainBatchNorm(nn.Module):
    """Per-domain BatchNorm: each domain has its own BN statistics.

    Args:
        num_features: Number of features (256 for bottleneck).
        num_domains: Number of source domains.
    """

    def __init__(self, num_features: int, num_domains: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features) for _ in range(num_domains)]
        )
        self.domain_to_idx: dict[str, int] = {}

    def forward(self, x: torch.Tensor, domain_indices: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, num_features]
            domain_indices: [B] integer domain indices.
        Returns:
            Normalized x [B, num_features].
        """
        output = torch.zeros_like(x)
        for d_idx in range(self.num_domains):
            mask = domain_indices == d_idx
            if mask.any():
                output[mask] = self.bns[d_idx](x[mask])
        return output

    def register_domain(self, domain_id: str, idx: int) -> None:
        self.domain_to_idx[domain_id] = idx

    def get_domain_idx(self, domain_id: str) -> int:
        return self.domain_to_idx.get(domain_id, 0)
