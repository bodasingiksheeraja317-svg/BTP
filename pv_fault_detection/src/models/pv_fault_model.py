"""
src/models/pv_fault_model.py
Top-level nn.Module: assembles all encoders, fusion, DG module, and heads per config.

This is the ONLY place architecture is assembled. Encoders/fusion/DG modules
are never instantiated ad hoc elsewhere in the codebase.

Batch dict contract (spec Section 9.4):
  Input keys:
    rgb:               Tensor [B, 3, 224, 224] or None
    thermal:           Tensor [B, 1or2, 224, 224] or None
    electrical_iv:     Tensor [B, 100, 2] or None
    electrical_pv:     Tensor [B, 100, 1] or None
    electrical_scalars: Tensor [B, 7] or None
    environment:       Tensor [B, 3] or None
    scalar_known_mask: Tensor [B, 7] bool or None
    env_known_mask:    Tensor [B, 3] bool or None
    domain_indices:    Tensor [B] int (for domain_norm DG method)
    domain_ids:        list[str] (for domain tracking)

  Output keys:
    fault_logits:   Tensor [B, 7]
    domain_logits:  Tensor [B, num_domains] or None
    z:              Tensor [B, 256]  (exposed for CORAL loss + visualization)

Spec reference: Section 9.4, Section 5, Section 10
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.models.encoders.electrical_encoder import ElectricalEncoder
from src.models.encoders.image_encoder import ImageEncoder
from src.models.fusion.cross_attention import GatedCrossAttention
from src.models.fusion.late_fusion import LateFusion, MultimodalLateFusion
from src.models.heads.classifier_head import ClassifierHead
from src.models.heads.domain_head import DomainHead

logger = logging.getLogger(__name__)


class PVFaultModel(nn.Module):
    """Top-level PV fault detection model.

    Assembles all components per config. Supports:
      - Any subset of modalities: [rgb], [thermal], [electrical], [rgb, thermal],
        [rgb, thermal, electrical] (spec Section 9.4)
      - Fusion types: cross_attention, late
      - DG methods: none, coral, adversarial, domain_norm
      - Missing-modality handling: learned substitute embeddings
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", {})

        self.modalities = model_cfg.get("modalities", ["rgb", "thermal", "electrical"])
        self.fusion_type = model_cfg.get("fusion_type", "cross_attention")
        self.dg_method = model_cfg.get("dg_method", "none")
        self.modality_dropout_prob = model_cfg.get("modality_dropout_prob", 0.15)

        self.rgb_dim = model_cfg.get("rgb_feature_dim", 256)
        self.thermal_dim = model_cfg.get("thermal_feature_dim", 256)
        self.elec_dim = model_cfg.get("elec_feature_dim", 128)
        self.shared_dim = model_cfg.get("shared_dim", 256)
        self.num_classes = model_cfg.get("num_fault_classes", 7)

        # --------------- Encoders (only for active modalities) ---------------
        if "rgb" in self.modalities:
            self.rgb_encoder = ImageEncoder(
                backbone=model_cfg.get("image_backbone", "efficientnet_b0"),
                pretrained=True,
                in_channels=3,
                out_dim=self.rgb_dim,
            )
        if "thermal" in self.modalities:
            thermal_channels = 2 if model_cfg.get("use_raw_radiometric", False) else 1
            self.thermal_encoder = ImageEncoder(
                backbone=model_cfg.get("image_backbone", "efficientnet_b0"),
                pretrained=True,
                in_channels=thermal_channels,
                out_dim=self.thermal_dim,
            )
        if "electrical" in self.modalities:
            self.electrical_encoder = ElectricalEncoder(
                transformer_d_model=model_cfg.get("elec_transformer_d_model", 128),
                transformer_nhead=model_cfg.get("elec_transformer_nhead", 4),
                transformer_layers=model_cfg.get("elec_transformer_layers", 2),
                n_scalars=model_cfg.get("elec_num_scalars", 7),
                n_env=model_cfg.get("elec_num_env", 3),
                out_dim=self.elec_dim,
                dropout=model_cfg.get("transformer_dropout", 0.1),
            )

        # --------------- Missing-modality embeddings ---------------
        if "rgb" in self.modalities:
            self.missing_rgb_embed = nn.Parameter(torch.randn(1, self.rgb_dim) * 0.02)
        if "thermal" in self.modalities:
            self.missing_thermal_embed = nn.Parameter(torch.randn(1, self.thermal_dim) * 0.02)
        if "electrical" in self.modalities:
            self.missing_elec_embed = nn.Parameter(torch.randn(1, self.elec_dim) * 0.02)

        # --------------- Fusion ---------------
        image_modalities = [m for m in self.modalities if m in ("rgb", "thermal")]

        if self.fusion_type == "cross_attention" and len(image_modalities) == 2:
            self.cross_attention = GatedCrossAttention(
                dim=self.shared_dim,
                num_heads=model_cfg.get("cross_attn_heads", 2),
            )
            img_out_dim = self.shared_dim
        else:
            img_dims = [self.rgb_dim if "rgb" in self.modalities else 0,
                        self.thermal_dim if "thermal" in self.modalities else 0]
            img_dims = [d for d in img_dims if d > 0]
            if len(img_dims) > 1:
                self.image_late_fusion = MultimodalLateFusion(
                    feature_dims=img_dims, out_dim=self.shared_dim
                )
            img_out_dim = self.shared_dim if img_dims else 0

        # --------------- Bottleneck (image + electrical -> z) ---------------
        if "electrical" in self.modalities and img_out_dim > 0:
            self.bottleneck = LateFusion(
                image_dim=img_out_dim,
                elec_dim=self.elec_dim,
                out_dim=self.shared_dim,
                dropout=model_cfg.get("bottleneck_dropout", 0.2),
            )
        elif "electrical" in self.modalities and img_out_dim == 0:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.elec_dim, self.shared_dim),
                nn.LayerNorm(self.shared_dim),
                nn.GELU(),
                nn.Dropout(model_cfg.get("bottleneck_dropout", 0.2)),
            )
        else:
            if img_out_dim != self.shared_dim:
                self.bottleneck = nn.Linear(img_out_dim, self.shared_dim)
            else:
                self.bottleneck = nn.Identity()

        # --------------- DG: Domain-specific normalization ---------------
        if self.dg_method == "domain_norm":
            from src.models.domain_generalization.domain_norm import DomainBatchNorm
            num_domains = cfg.get("num_domains", 6)
            self.domain_bn = DomainBatchNorm(
                num_features=self.shared_dim, num_domains=num_domains
            )

        # --------------- Heads ---------------
        self.classifier_head = ClassifierHead(
            in_dim=self.shared_dim,
            num_classes=self.num_classes,
            dropout=model_cfg.get("bottleneck_dropout", 0.2),
        )
        if self.dg_method == "adversarial":
            num_domains = cfg.get("num_domains", 6)
            self.domain_head = DomainHead(
                in_dim=self.shared_dim,
                num_domains=num_domains,
            )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward pass over a batch dict."""
        device = next(self.parameters()).device

        f_rgb = self._encode_rgb(batch, device)
        f_thermal = self._encode_thermal(batch, device)
        f_elec = self._encode_electrical(batch, device)

        f_img = self._fuse_images(f_rgb, f_thermal, batch)
        z = self._compute_z(f_img, f_elec, batch, device)

        if self.dg_method == "domain_norm":
            domain_indices = batch.get("domain_indices")
            if domain_indices is not None:
                z = self.domain_bn(z, domain_indices.to(device))

        fault_logits = self.classifier_head(z)

        domain_logits = None
        if self.dg_method == "adversarial" and self.training:
            domain_logits = self.domain_head(z)

        return {
            "fault_logits": fault_logits,
            "domain_logits": domain_logits,
            "z": z,
        }

    def _encode_rgb(self, batch, device) -> torch.Tensor | None:
        if "rgb" not in self.modalities:
            return None
        rgb = batch.get("rgb")
        has_rgb = batch.get("has_rgb")
        B = self._get_batch_size(batch)
        missing = self.missing_rgb_embed.expand(B, -1)

        if rgb is not None and (has_rgb is None or has_rgb.any()):
            feats = self.rgb_encoder(rgb.to(device))
            if has_rgb is not None:
                mask = has_rgb.to(device).unsqueeze(-1).float()  # [B, 1]
                if self.training and self.modality_dropout_prob > 0:
                    drop = (torch.rand(B, 1, device=device) >= self.modality_dropout_prob).float()
                    mask = mask * drop
                return feats * mask + missing * (1.0 - mask)
            else:
                if self.training and torch.rand(1).item() < self.modality_dropout_prob:
                    return missing
                return feats
        return missing

    def _encode_thermal(self, batch, device) -> torch.Tensor | None:
        if "thermal" not in self.modalities:
            return None
        thermal = batch.get("thermal")
        has_thermal = batch.get("has_thermal")
        B = self._get_batch_size(batch)
        missing = self.missing_thermal_embed.expand(B, -1)

        if thermal is not None and (has_thermal is None or has_thermal.any()):
            feats = self.thermal_encoder(thermal.to(device))
            if has_thermal is not None:
                mask = has_thermal.to(device).unsqueeze(-1).float()  # [B, 1]
                if self.training and self.modality_dropout_prob > 0:
                    drop = (torch.rand(B, 1, device=device) >= self.modality_dropout_prob).float()
                    mask = mask * drop
                return feats * mask + missing * (1.0 - mask)
            else:
                if self.training and torch.rand(1).item() < self.modality_dropout_prob:
                    return missing
                return feats
        return missing

    def _encode_electrical(self, batch, device) -> torch.Tensor | None:
        if "electrical" not in self.modalities:
            return None
        iv = batch.get("electrical_iv")
        pv = batch.get("electrical_pv")
        scalars = batch.get("electrical_scalars")
        env = batch.get("environment")
        has_elec = batch.get("has_electrical")
        B = self._get_batch_size(batch)
        missing = self.missing_elec_embed.expand(B, -1)

        if iv is None or pv is None:
            return missing
        if has_elec is not None and not has_elec.any():
            return missing

        B_cur = iv.shape[0]
        if scalars is None:
            scalars = torch.zeros(B_cur, 7, device=device)
        if env is None:
            env = torch.zeros(B_cur, 3, device=device)
        feats = self.electrical_encoder(
            iv.to(device), pv.to(device), scalars.to(device), env.to(device),
            scalar_known_mask=batch.get("scalar_known_mask"),
            env_known_mask=batch.get("env_known_mask"),
        )
        if has_elec is not None:
            mask = has_elec.to(device).unsqueeze(-1).float()
            return feats * mask + missing * (1.0 - mask)
        return feats

    def _fuse_images(self, f_rgb, f_thermal, batch) -> torch.Tensor | None:
        if f_rgb is None and f_thermal is None:
            return None
        if f_rgb is None:
            return f_thermal
        if f_thermal is None:
            return f_rgb
        if self.fusion_type == "cross_attention" and hasattr(self, "cross_attention"):
            return self.cross_attention(f_rgb, f_thermal)
        elif hasattr(self, "image_late_fusion"):
            return self.image_late_fusion([f_rgb, f_thermal])
        return f_rgb

    def _compute_z(self, f_img, f_elec, batch, device) -> torch.Tensor:
        if f_img is not None and f_elec is not None:
            if isinstance(self.bottleneck, LateFusion):
                return self.bottleneck(f_img, f_elec)
            return self.bottleneck(torch.cat([f_img, f_elec], dim=-1))
        elif f_img is not None:
            return self.bottleneck(f_img)
        elif f_elec is not None:
            return self.bottleneck(f_elec)
        else:
            raise RuntimeError("No modalities available for forward pass.")

    def _get_batch_size(self, batch) -> int:
        for v in batch.values():
            if v is not None and hasattr(v, "shape"):
                return v.shape[0]
        return 1


def build_model(cfg: dict) -> PVFaultModel:
    """Construct PVFaultModel from config dict."""
    return PVFaultModel(cfg)
