"""
tests/integration/test_raw_to_dataloader.py
Integration test: index DataFrame -> PVDataset -> DataLoader batch generation.
"""
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from src.data.dataset import PVDataset, pv_collate_fn

pytestmark = pytest.mark.integration


def test_raw_to_dataloader():
    """Verify PVDataset and DataLoader batch generation from Parquet DataFrame."""
    df = pd.DataFrame([
        {
            "sample_id": "test__1",
            "domain_id": "test_domain",
            "site_id": "site_1",
            "fault_label": "['Hotspot', 'Soiling']",
            "rgb_path": None,
            "thermal_path": None,
            "electrical_path": None,
            "environment": '{"irradiance_w_m2": 800.0, "ambient_temp_c": 25.0, "wind_m_s": 1.0}',
        },
        {
            "sample_id": "test__2",
            "domain_id": "test_domain",
            "site_id": "site_1",
            "fault_label": "['Healthy']",
            "rgb_path": None,
            "thermal_path": None,
            "electrical_path": None,
            "environment": None,
        }
    ])

    dataset = PVDataset(df, is_training=False)
    assert len(dataset) == 2

    loader = DataLoader(dataset, batch_size=2, collate_fn=pv_collate_fn)
    batch = next(iter(loader))

    assert "sample_id" in batch
    assert "fault_targets" in batch
    assert batch["fault_targets"].shape == (2, 7)
    # Check multi-label encoding
    assert batch["fault_targets"][0, 1] == 1.0  # Hotspot index is 1
    assert batch["fault_targets"][0, 4] == 1.0  # Soiling index is 4
    assert batch["fault_targets"][1, 0] == 1.0  # Healthy index is 0


def test_mixed_modality_batch_forward(tmp_path):
    """Verify mixed-modality batches: present modalities are encoded and absent ones get missing embeddings."""
    import cv2
    import numpy as np
    from src.common.config import load_config
    from src.models.pv_fault_model import build_model

    # Create dummy images
    img1_path = tmp_path / "rgb_sample1.png"
    dummy_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(img1_path), dummy_img)

    th2_path = tmp_path / "th_sample2.png"
    dummy_th = (np.random.rand(224, 224) * 255).astype(np.uint8)
    cv2.imwrite(str(th2_path), dummy_th)

    # Sample 1: RGB present, Thermal absent
    # Sample 2: RGB absent, Thermal present
    df = pd.DataFrame([
        {
            "sample_id": "sample_rgb_only",
            "domain_id": "domain_a",
            "site_id": "site_a",
            "fault_label": "['Healthy']",
            "rgb_path": str(img1_path),
            "thermal_path": None,
            "electrical_path": None,
            "environment": None,
        },
        {
            "sample_id": "sample_th_only",
            "domain_id": "domain_a",
            "site_id": "site_a",
            "fault_label": "['Hotspot']",
            "rgb_path": None,
            "thermal_path": str(th2_path),
            "electrical_path": None,
            "environment": None,
        },
    ])

    dataset = PVDataset(df, modalities=["rgb", "thermal"], is_training=False)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pv_collate_fn)
    batch = next(iter(loader))

    assert "has_rgb" in batch
    assert "has_thermal" in batch
    assert batch["has_rgb"].tolist() == [True, False]
    assert batch["has_thermal"].tolist() == [False, True]
    assert batch["rgb"].shape == (2, 3, 224, 224)
    assert batch["thermal"].shape == (2, 1, 224, 224)

    cfg = load_config("configs/multimodal_rgb_thermal.yaml", overrides=["hardware.device=cpu"])
    model = build_model(cfg)
    model.eval()

    with torch.no_grad():
        f_rgb = model._encode_rgb(batch, device=torch.device("cpu"))
        f_thermal = model._encode_thermal(batch, device=torch.device("cpu"))

        # Sample 1 has real RGB: f_rgb[0] != missing_rgb_embed
        # Sample 2 has missing RGB: f_rgb[1] == missing_rgb_embed
        diff_sample1_rgb = torch.norm(f_rgb[0] - model.missing_rgb_embed[0]).item()
        diff_sample2_rgb = torch.norm(f_rgb[1] - model.missing_rgb_embed[0]).item()
        assert diff_sample1_rgb > 1e-3, "Sample 1 should have real RGB features"
        assert diff_sample2_rgb < 1e-5, "Sample 2 should get missing RGB embedding"

        # Sample 1 has missing Thermal: f_thermal[0] == missing_thermal_embed
        # Sample 2 has real Thermal: f_thermal[1] != missing_thermal_embed
        diff_sample1_th = torch.norm(f_thermal[0] - model.missing_thermal_embed[0]).item()
        diff_sample2_th = torch.norm(f_thermal[1] - model.missing_thermal_embed[0]).item()
        assert diff_sample1_th < 1e-5, "Sample 1 should get missing Thermal embedding"
        assert diff_sample2_th > 1e-3, "Sample 2 should have real Thermal features"

        # Full forward pass
        out = model(batch)
        assert out["fault_logits"].shape == (2, 7)
        assert not torch.isnan(out["fault_logits"]).any()
