"""
src/models/encoders/image_encoder.py
EfficientNet-B0 / ConvNeXt-Tiny image encoder wrapper (timm-based).

Spec Section 10:
  - EfficientNet-B0, ImageNet pretrained, separate instances for RGB and thermal
  - Input: 224x224xC (C=3 for RGB, C=1 or 2 for thermal)
  - Output: 256-d feature vector (after GAP + linear projection head)
  - Thermal encoder first conv adapted for 1 or 2 input channels
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """EfficientNet-B0 or ConvNeXt-Tiny encoder with projection head.

    Args:
        backbone: 'efficientnet_b0' or 'convnext_tiny'.
        pretrained: If True, load ImageNet weights.
        in_channels: Number of input channels (3 for RGB, 1 or 2 for thermal).
        out_dim: Output feature dimension (256 per spec).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 3,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.out_dim = out_dim

        try:
            import timm
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,          # remove classification head
                global_pool="avg",       # GAP included
                in_chans=in_channels,
            )
            backbone_out_dim = self.backbone.num_features
        except ImportError:
            raise ImportError(
                "timm is required for ImageEncoder. Install: pip install timm"
            )

        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor [B, in_channels, 224, 224].

        Returns:
            Feature vector [B, out_dim=256].
        """
        features = self.backbone(x)           # [B, backbone_out_dim]
        projected = self.projection(features)  # [B, 256]
        return projected
