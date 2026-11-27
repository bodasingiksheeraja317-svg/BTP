"""
tests/integration/test_dataloader_to_model.py
Integration test: full pipeline test from DataLoader batch directly through PVFaultModel.
"""
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from src.common.config import load_config
from src.data.dataset import PVDataset, pv_collate_fn
from src.models.pv_fault_model import build_model

pytestmark = pytest.mark.integration


def test_dataloader_batch_to_model_forward():
    """Verify batch output from DataLoader feeds directly into PVFaultModel without errors."""
    df = pd.DataFrame([
        {
            "sample_id": "test__1",
            "domain_id": "test_domain",
            "site_id": "site_1",
            "fault_label": "['Hotspot']",
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

    cfg = load_config("configs/multimodal_full.yaml")
    model = build_model(cfg)
    model.eval()

    dataset = PVDataset(df, is_training=False)
    loader = DataLoader(dataset, batch_size=2, collate_fn=pv_collate_fn)
    batch = next(iter(loader))

    with torch.no_grad():
        outputs = model(batch)

    assert "fault_logits" in outputs
    assert "z" in outputs
    assert outputs["fault_logits"].shape == (2, 7)
    assert outputs["z"].shape == (2, 256)
    assert not torch.isnan(outputs["fault_logits"]).any()
