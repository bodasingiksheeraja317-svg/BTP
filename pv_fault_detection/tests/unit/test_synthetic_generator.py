"""
tests/unit/test_synthetic_generator.py
Unit tests for synthetic quality control, valid ranges, and fault injection tables.
"""
from pathlib import Path
import numpy as np
import pytest
import yaml
from src.synthetic.quality_control import (
    validate_electrical_curve,
    validate_synthetic_image_ssim,
)

pytestmark = pytest.mark.unit


def test_validate_electrical_curve_physical_ranges():
    """Verify physical range checking rejects unphysical curves."""
    # Realistic valid scalars for Si module
    valid_scalars = {"Voc": 38.0, "Isc": 9.2, "FF": 0.78}
    passes, reason = validate_electrical_curve(valid_scalars)
    assert passes is True
    assert reason == ""

    # Unphysical Voc (too high: 100V for single module)
    invalid_voc = {"Voc": 100.0, "Isc": 9.2, "FF": 0.78}
    passes_voc, reason_voc = validate_electrical_curve(invalid_voc)
    assert passes_voc is False
    assert "Voc" in reason_voc

    # Unphysical FF (too low: 0.2)
    invalid_ff = {"Voc": 38.0, "Isc": 9.2, "FF": 0.20}
    passes_ff, reason_ff = validate_electrical_curve(invalid_ff)
    assert passes_ff is False
    assert "FF" in reason_ff


def test_validate_synthetic_image_ssim():
    """Verify SSIM validation thresholding."""
    img1 = (np.random.rand(100, 100) * 255).astype(np.uint8)
    # Identical image -> SSIM = 1.0
    passes_same, score_same = validate_synthetic_image_ssim(img1, img1, ssim_threshold=0.75)
    assert passes_same is True
    assert score_same == pytest.approx(1.0, abs=1e-3)

    # Completely different random image -> SSIM < 0.75
    img2 = (np.random.rand(100, 100) * 255).astype(np.uint8)
    passes_diff, score_diff = validate_synthetic_image_ssim(img1, img2, ssim_threshold=0.75)
    assert passes_diff is False


def test_fault_injection_table_completeness():
    """Verify fault_injection_table.yaml defines rules for all 7 fault classes."""
    table_path = Path("src/synthetic/fault_injection_table.yaml")
    assert table_path.exists()

    with open(table_path) as f:
        data = yaml.safe_load(f)

    assert "fault_classes" in data
    fault_classes = data["fault_classes"]
    expected_classes = [
        "Healthy",
        "Hotspot",
        "Open/Short-circuit",
        "Shading/Obstruction",
        "Soiling",
        "Structural (visual)",
        "Electrical/Diode fault",
    ]
    for cls in expected_classes:
        assert cls in fault_classes, f"Missing fault class '{cls}' in fault_injection_table.yaml"


def test_generate_iv_curve_all_fault_classes():
    """Verify generate_iv_curve_for_fault generates valid, physical curves for all 7 fault classes."""
    from src.synthetic.electrical_sim import generate_iv_curve_for_fault

    classes = [
        "Healthy",
        "Hotspot",
        "Open/Short-circuit",
        "Shading/Obstruction",
        "Soiling",
        "Structural (visual)",
        "Electrical/Diode fault",
    ]

    for fault_cls in classes:
        res = generate_iv_curve_for_fault(fault_class=fault_cls, irradiance=1000.0, temperature=25.0)
        assert res is not None, f"Failed to generate curve for fault class {fault_cls}"
        assert "iv_curve" in res
        assert "pv_curve" in res
        assert "scalars" in res
        assert res["iv_curve"].shape == (100, 2)
        assert res["pv_curve"].shape == (100, 1)
        assert res["scalars"].shape == (7,)

        voc, isc, vmp, imp, ff, rs, rsh = res["scalars"]
        assert voc > 10.0, f"{fault_cls}: Voc too low ({voc})"
        assert isc > 0.1, f"{fault_cls}: Isc too low ({isc})"
        assert ff > 0.15, f"{fault_cls}: FF too low ({ff})"
        assert rs >= 0.0, f"{fault_cls}: Rs must be non-negative"
        assert rsh >= 1.0, f"{fault_cls}: Rsh must be positive"
