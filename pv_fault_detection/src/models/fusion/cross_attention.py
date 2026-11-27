"""
src/models/fusion/cross_attention.py
Gated bidirectional cross-attention for RGB-Thermal image fusion.

Spec Section 5 / Section 10:
  - Bidirectional: RGB attends to thermal, thermal attends to RGB
  - 2 attention heads, hidden 256
  - Gate = learned sigmoid scalar per sample controlling attention contribution
  - Degrades gracefully to identity when one modality is the missing-modality embedding
  - Output: f_img (256-d)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttention(nn.Module):
    """Gated bidirectional cross-attention for two feature vectors.

    Args:
        dim: Feature dimension (256).
        num_heads: Number of attention heads (2 per spec).
        dropout: Attention dropout.
    """

    def __init__(self, dim: int = 256, num_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Cross-attention: RGB attends to thermal
        self.rgb_to_thermal_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: thermal attends to RGB
        self.thermal_to_rgb_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Gating scalars (learned sigmoid per sample, per direction)
        self.gate_rgb = nn.Linear(dim * 2, 1)
        self.gate_thermal = nn.Linear(dim * 2, 1)

        # Output projection: combine gated bidirectional attention
        self.output_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(
        self,
        f_rgb: torch.Tensor,      # [B, 256]
        f_thermal: torch.Tensor,  # [B, 256]
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            f_img: [B, 256] fused image feature.
        """
        # Expand to sequence dim=1 for MultiheadAttention: [B, 1, 256]
        rgb_seq = f_rgb.unsqueeze(1)
        thermal_seq = f_thermal.unsqueeze(1)

        # RGB attends to thermal
        rgb_attended, _ = self.rgb_to_thermal_attn(
            query=rgb_seq, key=thermal_seq, value=thermal_seq
        )  # [B, 1, 256]
        rgb_attended = rgb_attended.squeeze(1)  # [B, 256]

        # Thermal attends to RGB
        thermal_attended, _ = self.thermal_to_rgb_attn(
            query=thermal_seq, key=rgb_seq, value=rgb_seq
        )  # [B, 1, 256]
        thermal_attended = thermal_attended.squeeze(1)  # [B, 256]

        # Gating: learned sigmoid scalar per sample
        gate_rgb = torch.sigmoid(self.gate_rgb(torch.cat([f_rgb, rgb_attended], dim=-1)))  # [B, 1]
        gate_thermal = torch.sigmoid(self.gate_thermal(torch.cat([f_thermal, thermal_attended], dim=-1)))  # [B, 1]

        # Gated combination: residual + attention, controlled by gate
        rgb_fused = f_rgb + gate_rgb * rgb_attended          # [B, 256]
        thermal_fused = f_thermal + gate_thermal * thermal_attended  # [B, 256]

        # Merge bidirectional features
        merged = torch.cat([rgb_fused, thermal_fused], dim=-1)  # [B, 512]
        f_img = self.output_proj(merged)                          # [B, 256]
        return f_img
