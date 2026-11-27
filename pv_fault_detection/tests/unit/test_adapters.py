"""
tests/unit/test_adapters.py
Unit tests for dataset adapters using fixture-scale data.
"""
from pathlib import Path
import pytest
from src.data.adapters.base import CANONICAL_FAULT_CLASSES, CanonicalSample, RawSample
from src.data.adapters.irsm import IRSMAdapter
from src.data.adapters.elpv import ELPVAdapter
from src.data.adapters.cali import CaliAdapter
from src.data.adapters.solarfcd import SolarFCDAdapter
from src.data.adapters.rgbirpv import RGBIRPVAdapter
from src.data.adapters.bimodal_uav import BimodalUAVAdapter
from src.data.adapters.nrel_pvdaq import NRELPVDAQAdapter
from src.data.adapters.own_site import OwnSiteAdapter
from src.data.validate import validate_canonical_sample

pytestmark = pytest.mark.unit


def test_missing_raw_directory_returns_false_no_crash(tmp_path):
    """Adapters should return validate_raw() == False gracefully if directory does not exist."""
    non_existent = tmp_path / "non_existent"
    adapters = [
        IRSMAdapter(raw_root=non_existent),
        ELPVAdapter(raw_root=non_existent),
        CaliAdapter(raw_root=non_existent),
        SolarFCDAdapter(raw_root=non_existent),
        RGBIRPVAdapter(raw_root=non_existent),
        BimodalUAVAdapter(raw_root=non_existent),
        NRELPVDAQAdapter(raw_root=non_existent),
        OwnSiteAdapter(site_id="own_site_test", raw_root=non_existent),
    ]
    for adapter in adapters:
        assert adapter.validate_raw() is False
        assert adapter.is_available() is False
        assert adapter.status() == "MISSING"


def test_irsm_adapter_with_fixture(tmp_path):
    """Verify IRSM adapter parses image directory structure, maps labels, and populates thermal_path."""
    from src.data.indexer import _process_media
    irsm_root = tmp_path / "irsm"
    (irsm_root / "hot-spot").mkdir(parents=True)
    img_file = irsm_root / "hot-spot" / "sample_001.jpg"
    img_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # mock jpeg header

    adapter = IRSMAdapter(raw_root=irsm_root)
    assert adapter.validate_raw() is True

    samples = list(adapter.iter_samples())
    assert len(samples) == 1
    raw = samples[0]
    assert raw.original_id == "sample_001"
    assert raw.raw_fields["native_label"] == "hot-spot"

    canonical = adapter.to_canonical(raw)
    assert canonical.sample_id == "irsm__sample_001"
    assert canonical.domain_id == "irsm"
    assert canonical.fault_label == ["Hotspot"]
    assert canonical.thermal_path is not None
    assert str(img_file) in canonical.thermal_path

    # Process media
    processed_canonical = _process_media(canonical, adapter_id="irsm")
    assert processed_canonical.thermal_path is not None
    assert Path(processed_canonical.thermal_path).exists()

    errors = validate_canonical_sample(processed_canonical)
    assert len(errors) == 0


def test_own_site_adapter_with_fixture(tmp_path):
    """Verify own_site adapter parses CSV, populates paths, and handles multi-label / environmental metadata."""
    site_root = tmp_path / "own_site_a"
    site_root.mkdir()
    csv_file = site_root / "labels.csv"
    csv_file.write_text(
        "panel_id,timestamp,fault_label,inspection_pass_id,irradiance_w_m2,ambient_temp_c,wind_m_s,rgb_path,thermal_path\n"
        "p01,2026-08-30T10:00:00Z,Hotspot|Soiling,flight_01,850.5,32.0,2.1,data/raw/own_site_a/p01_rgb.jpg,data/raw/own_site_a/p01_th.jpg\n"
    )

    adapter = OwnSiteAdapter(site_id="own_site_a", raw_root=site_root)
    assert adapter.validate_raw() is True

    samples = list(adapter.iter_samples())
    assert len(samples) == 1
    raw = samples[0]

    canonical = adapter.to_canonical(raw)
    assert canonical.sample_id == "own_site_a__p01_2026-08-30T10:00:00Z"
    assert canonical.domain_id == "own_site_a"
    assert set(canonical.fault_label) == {"Hotspot", "Soiling"}
    assert canonical.rgb_path == "data/raw/own_site_a/p01_rgb.jpg"
    assert canonical.thermal_path == "data/raw/own_site_a/p01_th.jpg"
    assert canonical.environment == {
        "irradiance_w_m2": 850.5,
        "ambient_temp_c": 32.0,
        "wind_m_s": 2.1,
    }

    errors = validate_canonical_sample(canonical)
    assert len(errors) == 0
