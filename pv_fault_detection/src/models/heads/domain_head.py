"""
src/models/heads/domain_head.py
Training-only domain classifier head with gradient reversal.

Spec Section 10:
  Linear(256 -> num_domains), softmax logits.
  Gradient-reversal layer upstream (standard DANN recipe).
  DISCARDED at inference — only present when model.dg_method == adversarial.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal layer (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal layer with configurable alpha (lambda)."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


class DomainHead(nn.Module):
    """Domain classifier head for adversarial DG (training only).

    Args:
        in_dim: Input dimension (256).
        num_domains: Number of source domains.
        alpha: Gradient-reversal scaling factor (scheduled externally).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_domains: int = 2,
        alpha: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.grl = GradientReversal(alpha=alpha)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_domains),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: Shared representation [B, 256].
        Returns:
            domain_logits: [B, num_domains] (raw logits).
        """
        z_reversed = self.grl(z)
        return self.head(z_reversed)

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)
