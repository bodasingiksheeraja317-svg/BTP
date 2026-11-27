"""
tests/leakage/test_no_panel_crosses_split.py
Leakage test: no panel_id or inspection_pass_id appears in more than one split partition.

This is a LOAD-BEARING test. If it fails, the pipeline splitting code is wrong, not this test.
"""
import pandas as pd
import pytest
from src.splitting.grouped_split import grouped_split

pytestmark = [pytest.mark.leakage]


def test_no_panel_crosses_split_boundary():
    """Verify that every panel_id belongs exclusively to exactly one split set."""
    # Generate multi-panel dataset with repeat visits / multi-angles
    records = []
    for panel_idx in range(50):
        for rep in range(4):  # 4 shots of the same physical panel
            records.append({
                "sample_id": f"site1__p{panel_idx}_shot{rep}",
                "domain_id": "site1",
                "site_id": "site1",
                "panel_id": f"panel_{panel_idx}",
                "inspection_pass_id": f"pass_{rep // 2}",
                "fault_label": "['Healthy']",
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = grouped_split(df, train_frac=0.7, val_frac=0.15, seed=123)

    train_panels = set(train_df["panel_id"])
    val_panels = set(val_df["panel_id"])
    test_panels = set(test_df["panel_id"])

    # Strict pairwise disjointness
    train_val_overlap = train_panels.intersection(val_panels)
    train_test_overlap = train_panels.intersection(test_panels)
    val_test_overlap = val_panels.intersection(test_panels)

    assert len(train_val_overlap) == 0, f"DATA LEAKAGE: panel_id in both train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"DATA LEAKAGE: panel_id in both train and test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"DATA LEAKAGE: panel_id in both val and test: {val_test_overlap}"


def test_no_inspection_pass_crosses_when_panel_null():
    """When panel_id is null, inspection_pass_id must be pairwise disjoint across splits."""
    records = []
    for pass_idx in range(30):
        for shot in range(5):
            records.append({
                "sample_id": f"site2__pass{pass_idx}_shot{shot}",
                "domain_id": "site2",
                "site_id": "site2",
                "panel_id": None,
                "inspection_pass_id": f"pass_{pass_idx}",
                "fault_label": "['Hotspot']",
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = grouped_split(df, train_frac=0.6, val_frac=0.2, seed=42)

    train_passes = set(train_df["inspection_pass_id"])
    val_passes = set(val_df["inspection_pass_id"])
    test_passes = set(test_df["inspection_pass_id"])

    assert len(train_passes.intersection(val_passes)) == 0, "DATA LEAKAGE: inspection_pass_id in train & val"
    assert len(train_passes.intersection(test_passes)) == 0, "DATA LEAKAGE: inspection_pass_id in train & test"
    assert len(val_passes.intersection(test_passes)) == 0, "DATA LEAKAGE: inspection_pass_id in val & test"
