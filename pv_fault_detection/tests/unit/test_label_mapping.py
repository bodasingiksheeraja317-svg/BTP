"""
tests/unit/test_label_mapping.py
Unit tests to ensure every YAML label map only maps to the canonical 7-class taxonomy.
"""
from pathlib import Path
import yaml
import pytest
from src.data.adapters.base import CANONICAL_FAULT_CLASSES

pytestmark = pytest.mark.unit


def test_canonical_fault_classes_count():
    """Canonical taxonomy must have exactly 7 classes per Section 7.1."""
    assert len(CANONICAL_FAULT_CLASSES) == 7
    expected = [
        "Healthy",
        "Hotspot",
        "Open/Short-circuit",
        "Shading/Obstruction",
        "Soiling",
        "Structural (visual)",
        "Electrical/Diode fault",
    ]
    assert CANONICAL_FAULT_CLASSES == expected


def test_all_yaml_label_maps_target_canonical_classes():
    """All non-TO_VERIFY mapped targets in src/data/label_maps/*.yaml must be valid canonical classes."""
    label_map_dir = Path("src/data/label_maps")
    yaml_files = list(label_map_dir.glob("*.yaml"))
    assert len(yaml_files) >= 7

    for yf in yaml_files:
        with open(yf) as f:
            mapping = yaml.safe_load(f) or {}
        for source_lbl, target_lbl in mapping.items():
            if target_lbl == "TO_VERIFY":
                continue
            if isinstance(target_lbl, list):
                for t in target_lbl:
                    assert t in CANONICAL_FAULT_CLASSES, (
                        f"In {yf.name}: mapped target '{t}' for '{source_lbl}' "
                        f"not in CANONICAL_FAULT_CLASSES"
                    )
            else:
                assert target_lbl in CANONICAL_FAULT_CLASSES, (
                    f"In {yf.name}: mapped target '{target_lbl}' for '{source_lbl}' "
                    f"not in CANONICAL_FAULT_CLASSES"
                )
