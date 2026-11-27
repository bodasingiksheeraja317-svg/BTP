"""
src/models/fusion/late_fusion.py
Late (concatenation) fusion for:
  1. Electrical branch + image features -> shared bottleneck z
  2. Ablation baseline: late fusion of all modalities (ablation row 6)

Spec Section 10:
  MLP: Linear(384 -> 256) -> LayerNorm -> GELU -> Dropout(0.2)
  f_img (256) + f_elec (128) -> Linear(384->256) -> z (256)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Concatenation + MLP bottleneck for combining image and electrical features.

    Args:
        image_dim: Dimension of image feature (256).
        elec_dim: Dimension of electrical feature (128).
        out_dim: Output shared representation dimension (256).
        dropout: Dropout probability (0.2 per spec).
    """

    def __init__(
        self,
        image_dim: int = 256,
        elec_dim: int = 128,
        out_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = image_dim + elec_dim  # 384
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, f_img: torch.Tensor, f_elec: torch.Tensor) -> torch.Tensor:
        """Args:
            f_img: Image features [B, 256].
            f_elec: Electrical features [B, 128].
        Returns:
            z: Shared representation [B, 256].
        """
        combined = torch.cat([f_img, f_elec], dim=-1)  # [B, 384]
        return self.bottleneck(combined)                 # [B, 256]


class MultimodalLateFusion(nn.Module):
    """Late-fusion baseline that concatenates all modality features directly.
    
    Used in ablation row 6 (late vs cross-attention comparison).
    Input can be any subset of [rgb, thermal, electrical] features.
    """

    def __init__(self, feature_dims: list[int], out_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        in_dim = sum(feature_dims)
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        combined = torch.cat(features, dim=-1)
        return self.bottleneck(combined)
