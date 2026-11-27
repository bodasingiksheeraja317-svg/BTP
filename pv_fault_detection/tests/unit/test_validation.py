"""
tests/unit/test_validation.py
Unit tests for canonical sample and Parquet row schema validation.
"""
import pytest
from src.data.adapters.base import CanonicalSample
from src.data.validate import (
    validate_canonical_sample,
    validate_parquet_row,
)

pytestmark = pytest.mark.unit


def test_valid_canonical_sample():
    """Well-formed sample passes validation."""
    sample = CanonicalSample(
        sample_id="domain_a__001",
        domain_id="domain_a",
        site_id="site_1",
        fault_label=["Hotspot", "Soiling"],
        panel_id="panel_42",
        environment={"irradiance_w_m2": 950.0, "ambient_temp_c": 28.0, "wind_m_s": 1.5},
        metadata={"license": "MIT", "dataset": "domain_a"},
    )
    errors = validate_canonical_sample(sample)
    assert len(errors) == 0


def test_sample_id_missing_separator_fails():
    """sample_id without '__' fails validation."""
    sample = CanonicalSample(
        sample_id="domain_a_001",  # single underscore
        domain_id="domain_a",
        site_id="site_1",
        fault_label=["Healthy"],
        metadata={"license": "MIT"},
    )
    errors = validate_canonical_sample(sample)
    assert any("must contain '__' separator" in e for e in errors)


def test_invalid_fault_class_fails():
    """Unknown fault label fails validation upon CanonicalSample construction."""
    with pytest.raises(AssertionError) as excinfo:
        CanonicalSample(
            sample_id="domain_a__001",
            domain_id="domain_a",
            site_id="site_1",
            fault_label=["InvalidFaultName"],
            metadata={"license": "MIT"},
        )
    assert "Unknown fault label 'InvalidFaultName'" in str(excinfo.value)


def test_validate_parquet_row():
    """Test Parquet row dictionary validation."""
    valid_row = {
        "sample_id": "test__1",
        "domain_id": "test",
        "site_id": "test",
        "fault_label": '["Healthy"]',
    }
    assert len(validate_parquet_row(valid_row)) == 0

    invalid_row = {
        "sample_id": "test__1",
        "domain_id": "test",
        "site_id": "test",
        "fault_label": '["UnknownClass"]',
    }
    assert len(validate_parquet_row(invalid_row)) > 0
