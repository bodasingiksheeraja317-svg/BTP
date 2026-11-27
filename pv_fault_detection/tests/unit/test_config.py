"""
tests/unit/test_config.py
Unit tests for configuration loading, hierarchical merge, and cross-key validation.
"""
from pathlib import Path
import pytest
from src.common.config import load_config, ConfigValidationError, _deep_merge, _apply_override

pytestmark = pytest.mark.unit


def test_base_config_loads():
    """Verify base.yaml loads cleanly and passes all cross-key validations."""
    cfg = load_config("configs/base.yaml")
    assert cfg is not None
    assert "model" in cfg
    assert "training" in cfg
    assert "data" in cfg
    assert "evaluation" in cfg


def test_all_configs_load():
    """Verify every configuration file in configs/ loads without validation errors."""
    config_files = list(Path("configs").glob("*.yaml"))
    assert len(config_files) == 10
    for path in config_files:
        cfg = load_config(path)
        assert cfg is not None


def test_dg_with_random_split_rejected():
    """Rule check: dg_method != 'none' with protocol: random_split must be rejected."""
    with pytest.raises(ConfigValidationError) as excinfo:
        load_config(
            "configs/base.yaml",
            overrides=["model.dg_method=coral", "evaluation.protocol=random_split"],
        )
    assert "requires evaluation.protocol='lodo'" in str(excinfo.value)


def test_invalid_synthetic_ratio_rejected():
    """Rule check: invalid synthetic ratio is rejected."""
    with pytest.raises(ConfigValidationError):
        load_config(
            "configs/base.yaml",
            overrides=["synthetic.ratio=999.0"],
        )


def test_deep_merge():
    """Verify deep merge logic."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 4}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
