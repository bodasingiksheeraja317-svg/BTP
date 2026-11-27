"""
tests/unit/test_fusion.py
Unit tests for cross-attention and late fusion output shape and gradient flow.
"""
import pytest
import torch
from src.models.fusion.cross_attention import GatedCrossAttention
from src.models.fusion.late_fusion import LateFusion, MultimodalLateFusion

pytestmark = pytest.mark.unit


def test_cross_attention_output_shape_and_gradient():
    """Verify GatedCrossAttention output shape and gradient backpropagation to both inputs."""
    fusion = GatedCrossAttention(dim=256, num_heads=2)
    f_rgb = torch.randn(4, 256, requires_grad=True)
    f_thermal = torch.randn(4, 256, requires_grad=True)

    f_img = fusion(f_rgb, f_thermal)
    assert f_img.shape == (4, 256)

    loss = f_img.sum()
    loss.backward()

    assert f_rgb.grad is not None
    assert f_thermal.grad is not None
    assert not torch.isnan(f_rgb.grad).any()
    assert not torch.isnan(f_thermal.grad).any()


def test_late_fusion_output_shape_and_gradient():
    """Verify LateFusion bottleneck output shape [B, 256] and gradient flow."""
    fusion = LateFusion(image_dim=256, elec_dim=128, out_dim=256)
    f_img = torch.randn(4, 256, requires_grad=True)
    f_elec = torch.randn(4, 128, requires_grad=True)

    z = fusion(f_img, f_elec)
    assert z.shape == (4, 256)

    loss = z.sum()
    loss.backward()

    assert f_img.grad is not None
    assert f_elec.grad is not None


def test_multimodal_late_fusion_baseline():
    """Verify MultimodalLateFusion baseline for arbitrary feature combinations."""
    fusion = MultimodalLateFusion(feature_dims=[256, 256, 128], out_dim=256)
    f1 = torch.randn(2, 256)
    f2 = torch.randn(2, 256)
    f3 = torch.randn(2, 128)

    z = fusion([f1, f2, f3])
    assert z.shape == (2, 256)
