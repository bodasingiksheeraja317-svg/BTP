"""
src/models/encoders/electrical_encoder.py
1D-CNN + Transformer encoder for I-V/P-V electrical curves.

Spec Section 10 / Section 5:
  Architecture:
    1D-CNN: 3 conv blocks, channels 2->32->64->128
    Transformer: 2 layers, d_model=128, nhead=4, over 100 curve tokens
    Output: 128-d sequence pooling -> concatenate scalar (7-d) + env (3-d) features
            -> MLP projection -> 128-d f_elec

  Trained from scratch (no pretrained electrical backbone exists).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ElectricalEncoder(nn.Module):
    """1D-CNN + Transformer encoder for electrical curves.

    Args:
        iv_channels: Number of input channels for IV curve (2: voltage, current).
        pv_channels: Number of P-V channels (1: power).
        cnn_channels: List of CNN channel sizes [in, h1, h2, h3].
        transformer_d_model: Transformer hidden dim.
        transformer_nhead: Number of attention heads.
        transformer_layers: Number of transformer encoder layers.
        n_scalars: Number of scalar features (7: Voc, Isc, Vmp, Imp, FF, Rs, Rsh).
        n_env: Number of environment features (3: irradiance, temp, wind).
        out_dim: Output dimension (128 per spec).
        dropout: Transformer dropout (0.1 per spec ASSUMPTION).
    """

    def __init__(
        self,
        iv_channels: int = 2,
        pv_channels: int = 1,
        cnn_channels: list = None,
        transformer_d_model: int = 128,
        transformer_nhead: int = 4,
        transformer_layers: int = 2,
        n_scalars: int = 7,
        n_env: int = 3,
        out_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [2, 32, 64, 128]  # per spec Section 10

        self.out_dim = out_dim
        total_curve_channels = iv_channels + pv_channels  # 3 = 2 (IV) + 1 (PV)
        cnn_in = total_curve_channels

        # 1D-CNN: 3 conv blocks
        cnn_layers = []
        prev_ch = cnn_in
        for ch in cnn_channels[1:]:
            cnn_layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ])
            prev_ch = ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Project to transformer_d_model if different
        cnn_out_ch = cnn_channels[-1]
        if cnn_out_ch != transformer_d_model:
            self.cnn_proj = nn.Linear(cnn_out_ch, transformer_d_model)
        else:
            self.cnn_proj = nn.Identity()

        # 2-layer Transformer encoder over 100 curve tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, transformer_d_model) * 0.02)

        # Scalar + env projection MLP
        scalar_env_dim = n_scalars + n_env  # 7 + 3 = 10
        self.scalar_env_proj = nn.Sequential(
            nn.Linear(scalar_env_dim, transformer_d_model),
            nn.GELU(),
        )

        # Missing-modality embeddings for null scalar/env values
        self.unknown_scalar_embed = nn.Parameter(torch.randn(n_scalars) * 0.02)
        self.unknown_env_embed = nn.Parameter(torch.randn(n_env) * 0.02)

        # Final projection: transformer_out + scalar_env_proj -> out_dim
        self.output_proj = nn.Sequential(
            nn.Linear(transformer_d_model * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        iv_curve: torch.Tensor,                    # [B, 100, 2]
        pv_curve: torch.Tensor,                    # [B, 100, 1]
        scalars: torch.Tensor,                     # [B, 7]
        environment: torch.Tensor,                 # [B, 3]
        scalar_known_mask: torch.Tensor | None = None,  # [B, 7] bool
        env_known_mask: torch.Tensor | None = None,     # [B, 3] bool
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            f_elec: [B, 128]
        """
        B = iv_curve.shape[0]

        # Concatenate IV + PV curves along channel dim: [B, 100, 3] -> [B, 3, 100]
        curve = torch.cat([iv_curve, pv_curve], dim=-1)  # [B, 100, 3]
        curve = curve.permute(0, 2, 1)                   # [B, 3, 100]

        # 1D-CNN
        cnn_out = self.cnn(curve)                        # [B, 128, 100]
        cnn_out = cnn_out.permute(0, 2, 1)              # [B, 100, 128]
        cnn_out = self.cnn_proj(cnn_out)                 # [B, 100, d_model]

        # Positional encoding
        cnn_out = cnn_out + self.pos_embed

        # Transformer
        transformer_out = self.transformer(cnn_out)      # [B, 100, d_model]
        pooled = transformer_out.mean(dim=1)             # [B, d_model] (mean pooling)

        # Handle unknown scalar/env values with learned embeddings
        scalars_out = self._apply_unknown_embed(
            scalars, scalar_known_mask, self.unknown_scalar_embed
        )
        env_out = self._apply_unknown_embed(
            environment, env_known_mask, self.unknown_env_embed
        )

        # Concatenate and project scalar+env
        scalar_env = torch.cat([scalars_out, env_out], dim=-1)  # [B, 10]
        scalar_env_feat = self.scalar_env_proj(scalar_env)       # [B, d_model]

        # Combine transformer output with scalar/env features
        combined = torch.cat([pooled, scalar_env_feat], dim=-1)  # [B, d_model*2]
        f_elec = self.output_proj(combined)                       # [B, 128]
        return f_elec

    def _apply_unknown_embed(
        self,
        values: torch.Tensor,
        known_mask: torch.Tensor | None,
        embed: nn.Parameter,
    ) -> torch.Tensor:
        """Substitute learned unknown embedding for null values."""
        if known_mask is None:
            return values
        embed_expanded = embed.unsqueeze(0).expand(values.shape[0], -1)
        mask = known_mask.float().to(values.device)
        return values * mask + embed_expanded * (1.0 - mask)
