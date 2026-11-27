"""
src/models/heads/classifier_head.py
Multi-label fault classification head.

Spec Section 10:
  Linear(256 -> 7), multi-label sigmoid.
  Class-weighted / focal loss (in losses.py, not here).
  Dropout 0.2 (spec ASSUMPTION).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Multi-label fault classification head.

    Args:
        in_dim: Input dimension (256 = shared bottleneck z).
        num_classes: Number of fault classes (7 per spec Section 7.1).
        dropout: Dropout probability (0.2 per spec ASSUMPTION).
    """

    def __init__(self, in_dim: int = 256, num_classes: int = 7, dropout: float = 0.2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args:
            z: Shared representation [B, 256].
        Returns:
            logits: [B, num_classes] (raw logits, apply sigmoid for probabilities).
        """
        return self.head(z)
