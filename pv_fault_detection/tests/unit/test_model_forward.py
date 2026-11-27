"""
tests/unit/test_model_forward.py
Unit tests for PVFaultModel forward pass across all modality configurations and missing-modality cases.
"""
import pytest
import torch
from src.common.config import load_config
from src.models.pv_fault_model import build_model

pytestmark = pytest.mark.unit


def _create_mock_batch(batch_size: int = 4):
    """Create a mock batch with all possible modality tensors."""
    return {
        "rgb": torch.randn(batch_size, 3, 224, 224),
        "thermal": torch.randn(batch_size, 1, 224, 224),
        "electrical_iv": torch.randn(batch_size, 100, 2),
        "electrical_pv": torch.randn(batch_size, 100, 1),
        "electrical_scalars": torch.randn(batch_size, 7),
        "environment": torch.randn(batch_size, 3),
        "scalar_known_mask": torch.ones(batch_size, 7, dtype=torch.bool),
        "env_known_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "fault_targets": torch.randint(0, 2, (batch_size, 7)).float(),
    }


def test_baseline_rgb_forward():
    """Verify forward pass on RGB-only baseline model."""
    cfg = load_config("configs/baseline_rgb.yaml")
    model = build_model(cfg)
    model.eval()

    batch = _create_mock_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)

    assert "fault_logits" in out
    assert "z" in out
    assert out["fault_logits"].shape == (2, 7)
    assert out["z"].shape == (2, 256)


def test_baseline_thermal_forward():
    """Verify forward pass on Thermal-only baseline model."""
    cfg = load_config("configs/baseline_thermal.yaml")
    model = build_model(cfg)
    model.eval()

    batch = _create_mock_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)

    assert out["fault_logits"].shape == (2, 7)
    assert out["z"].shape == (2, 256)


def test_baseline_electrical_forward():
    """Verify forward pass on Electrical-only baseline model."""
    cfg = load_config("configs/baseline_electrical.yaml")
    model = build_model(cfg)
    model.eval()

    batch = _create_mock_batch(batch_size=3)
    with torch.no_grad():
        out = model(batch)

    assert out["fault_logits"].shape == (3, 7)
    assert out["z"].shape == (3, 256)


def test_bimodal_rgb_thermal_forward():
    """Verify forward pass on Bimodal RGB+Thermal model with cross-attention."""
    cfg = load_config("configs/multimodal_rgb_thermal.yaml")
    model = build_model(cfg)
    model.eval()

    batch = _create_mock_batch(batch_size=2)
    with torch.no_grad():
        out = model(batch)

    assert out["fault_logits"].shape == (2, 7)
    assert out["z"].shape == (2, 256)


def test_trimodal_full_forward():
    """Verify forward pass on full trimodal (RGB+Thermal+Electrical) model."""
    cfg = load_config("configs/multimodal_full.yaml")
    model = build_model(cfg)
    model.eval()

    batch = _create_mock_batch(batch_size=4)
    with torch.no_grad():
        out = model(batch)

    assert out["fault_logits"].shape == (4, 7)
    assert out["z"].shape == (4, 256)


def test_missing_modality_fallback():
    """Verify trimodal model degrades gracefully when one modality is None."""
    cfg = load_config("configs/multimodal_full.yaml")
    model = build_model(cfg)
    model.eval()

    # Batch with no electrical data
    batch_no_elec = {
        "rgb": torch.randn(2, 3, 224, 224),
        "thermal": torch.randn(2, 1, 224, 224),
        "electrical_iv": None,
        "electrical_pv": None,
    }
    with torch.no_grad():
        out = model(batch_no_elec)

    assert out["fault_logits"].shape == (2, 7)
    assert out["z"].shape == (2, 256)
    assert not torch.isnan(out["fault_logits"]).any()
