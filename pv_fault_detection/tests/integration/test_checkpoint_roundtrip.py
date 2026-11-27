"""
tests/integration/test_checkpoint_roundtrip.py
Integration test: save model checkpoint -> load into fresh instance -> verify identical forward outputs.
"""
import pytest
import torch
from src.common.config import load_config
from src.models.pv_fault_model import build_model

pytestmark = pytest.mark.integration


def test_checkpoint_save_load_roundtrip(tmp_path):
    """Verify that saving and reloading a checkpoint produces identical forward outputs."""
    cfg = load_config("configs/multimodal_full.yaml")
    model_orig = build_model(cfg)
    model_orig.eval()

    batch = {
        "rgb": torch.randn(2, 3, 224, 224),
        "thermal": torch.randn(2, 1, 224, 224),
        "electrical_iv": torch.randn(2, 100, 2),
        "electrical_pv": torch.randn(2, 100, 1),
        "electrical_scalars": torch.randn(2, 7),
        "environment": torch.randn(2, 3),
    }

    with torch.no_grad():
        out_orig = model_orig(batch)

    # Save state dict
    ckpt_file = tmp_path / "model_ckpt.pt"
    torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_file)

    # Load into brand new model instance
    model_loaded = build_model(cfg)
    checkpoint = torch.load(ckpt_file, weights_only=False)
    model_loaded.load_state_dict(checkpoint["model_state_dict"])
    model_loaded.eval()

    with torch.no_grad():
        out_loaded = model_loaded(batch)

    # Verify identical outputs
    torch.testing.assert_close(out_orig["fault_logits"], out_loaded["fault_logits"])
    torch.testing.assert_close(out_orig["z"], out_loaded["z"])
