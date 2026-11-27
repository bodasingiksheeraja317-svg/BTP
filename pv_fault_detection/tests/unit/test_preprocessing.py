"""
tests/unit/test_preprocessing.py
Unit tests for RGB, Thermal, Environment, Synchronization, and Quality Control preprocessing.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.preprocessing.rgb import load_and_preprocess as preprocess_rgb
from src.preprocessing.thermal import load_and_preprocess as preprocess_thermal
from src.preprocessing.environment import process_environment
from src.preprocessing.synchronization import match_timestamps
from src.preprocessing.quality_control import passes_quality_control, check_thermal_image_quality

pytestmark = pytest.mark.unit


def test_rgb_preprocessing_pipeline(tmp_path):
    """Test RGB loading, resizing to 224x224x3, normalization, and blur detection."""
    # Create valid synthetic image
    img = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    img_path = tmp_path / "test_rgb.jpg"
    cv2.imwrite(str(img_path), img)

    out = preprocess_rgb(img_path, image_size=224, blur_filter_threshold=10.0)
    assert out is not None
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32

    # Create solid blank image (should be rejected by blur filter)
    blank_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    blank_path = tmp_path / "blank_rgb.jpg"
    cv2.imwrite(str(blank_path), blank_img)
    rejected_out = preprocess_rgb(blank_path, image_size=224, blur_filter_threshold=50.0)
    assert rejected_out is None


def test_thermal_preprocessing_pipeline(tmp_path):
    """Test Thermal loading, resizing, emissivity correction, and raw radiometric channel."""
    img = np.random.randint(20, 240, (100, 100), dtype=np.uint8)
    thermal_path = tmp_path / "test_thermal.jpg"
    cv2.imwrite(str(thermal_path), img)

    # 1-channel mode
    out_1ch = preprocess_thermal(thermal_path, image_size=224, use_raw_radiometric=False)
    assert out_1ch is not None
    assert out_1ch.shape == (1, 224, 224)

    # 2-channel mode with .npy raw temperatures
    raw_temp = np.random.uniform(20.0, 75.0, (100, 100)).astype(np.float32)
    raw_path = tmp_path / "test_raw.npy"
    np.save(str(raw_path), raw_temp)

    out_2ch = preprocess_thermal(
        thermal_path,
        thermal_raw_path=raw_path,
        image_size=224,
        use_raw_radiometric=True,
        emissivity=0.95,
        reflected_temp_c=25.0,
    )
    assert out_2ch is not None
    assert out_2ch.shape == (2, 224, 224)


def test_environment_cleaning_and_mask():
    """Test environment processing, out-of-range filtering, and learned unknown mask."""
    # Valid environment
    env_valid = {"irradiance_w_m2": 800.0, "ambient_temp_c": 25.0, "wind_m_s": 2.5}
    vals, mask = process_environment(env_valid)
    assert vals.shape == (3,)
    assert mask.all()

    # Partial / null environment
    env_partial = {"irradiance_w_m2": 800.0, "ambient_temp_c": None, "wind_m_s": None}
    vals, mask = process_environment(env_partial)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    assert bool(mask[2]) is False

    # Out of range irradiance
    env_invalid = {"irradiance_w_m2": 5000.0, "ambient_temp_c": 25.0, "wind_m_s": 2.5}
    vals, mask = process_environment(env_invalid)
    assert bool(mask[0]) is False  # Rejected as out of physical range (0-1500)


def test_timestamp_synchronization():
    """Test UTC timestamp matching within 120s tolerance."""
    images = [
        {"sample_id": "img1", "timestamp": "2026-08-30T12:00:00Z"},
        {"sample_id": "img2", "timestamp": "2026-08-30T12:10:00Z"},
    ]
    electrical = [
        {"electrical_path": "elec_001.npz", "timestamp": "2026-08-30T12:01:00Z"},  # 60s delta -> matches img1
        {"electrical_path": "elec_002.npz", "timestamp": "2026-08-30T12:20:00Z"},  # 600s delta -> too far
    ]

    matched = match_timestamps(images, electrical, max_delta_seconds=120)
    assert len(matched) == 2
    assert matched[0]["electrical_path"] == "elec_001.npz"
    assert matched[1]["electrical_path"] is None


def test_quality_control_checks():
    """Test IEC 62446-3 environmental quality checks."""
    passes, reason = passes_quality_control(irradiance=800.0, wind_speed=2.0)
    assert passes is True
    assert reason == ""

    # Low irradiance (< 600)
    passes_low, reason_low = passes_quality_control(irradiance=450.0, wind_speed=2.0)
    assert passes_low is False
    assert "irradiance" in reason_low

    # High wind (>= 5.0)
    passes_wind, reason_wind = passes_quality_control(irradiance=800.0, wind_speed=6.5)
    assert passes_wind is False
    assert "wind" in reason_wind
