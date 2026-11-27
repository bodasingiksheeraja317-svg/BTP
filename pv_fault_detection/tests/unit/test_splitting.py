"""
tests/unit/test_splitting.py
Unit tests for data splitting logic (random, grouped, site_held_out, and LODO).
"""
import pandas as pd
import pytest

from src.splitting.grouped_split import grouped_split
from src.splitting.lodo_split import generate_lodo_splits
from src.splitting.random_split import random_split
from src.splitting.site_held_out_split import site_held_out_split

pytestmark = pytest.mark.unit


def _create_mock_index():
    """Create a mock DataFrame with multiple sites, panels, passes, and samples."""
    rows = []
    # Site A (Domain A): 5 panels, 2 images per panel
    for p in range(1, 6):
        for img in [1, 2]:
            rows.append({
                "sample_id": f"site_a__p{p}_{img}",
                "domain_id": "domain_a",
                "site_id": "site_a",
                "panel_id": f"p{p}",
                "inspection_pass_id": "pass_1",
                "fault_label": "['Healthy']",
            })
    # Site B (Domain B): 5 panels
    for p in range(1, 6):
        for img in [1, 2]:
            rows.append({
                "sample_id": f"site_b__p{p}_{img}",
                "domain_id": "domain_b",
                "site_id": "site_b",
                "panel_id": f"p{p}_b",
                "inspection_pass_id": "pass_2",
                "fault_label": "['Hotspot']",
            })
    # Site C (Domain C): 5 panels
    for p in range(1, 6):
        for img in [1, 2]:
            rows.append({
                "sample_id": f"site_c__p{p}_{img}",
                "domain_id": "domain_c",
                "site_id": "site_c",
                "panel_id": f"p{p}_c",
                "inspection_pass_id": "pass_3",
                "fault_label": "['Soiling']",
            })
    return pd.DataFrame(rows)


def test_grouped_split_prevents_panel_overlap():
    """Verify grouped split never assigns any panel_id to multiple splits."""
    df = _create_mock_index()
    train_df, val_df, test_df = grouped_split(df, train_frac=0.6, val_frac=0.2, seed=42)

    train_panels = set(train_df["panel_id"].dropna())
    val_panels = set(val_df["panel_id"].dropna())
    test_panels = set(test_df["panel_id"].dropna())

    assert len(train_panels.intersection(val_panels)) == 0
    assert len(train_panels.intersection(test_panels)) == 0
    assert len(val_panels.intersection(test_panels)) == 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_site_held_out_split():
    """Verify site_held_out_split isolates the held-out domain completely into test set."""
    df = _create_mock_index()
    train_df, val_df, test_df = site_held_out_split(df, held_out_domain="domain_b", seed=42)

    assert set(test_df["domain_id"].unique()) == {"domain_b"}
    assert "domain_b" not in train_df["domain_id"].values
    assert "domain_b" not in val_df["domain_id"].values
    assert len(test_df) == 10  # 5 panels * 2 images


def test_lodo_split_generates_all_rotations(tmp_path):
    """Verify LODO generator produces all rotation files with zero leakage."""
    df = _create_mock_index()
    rotations = generate_lodo_splits(df, output_dir=tmp_path, seed=42)

    assert len(rotations) == 3
    assert set(rotations.keys()) == {"domain_a", "domain_b", "domain_c"}

    for domain_id, paths in rotations.items():
        assert paths["train"].exists()
        assert paths["val"].exists()
        assert paths["test"].exists()

        train_data = pd.read_parquet(paths["train"])
        val_data = pd.read_parquet(paths["val"])
        test_data = pd.read_parquet(paths["test"])

        # Leakage checks
        assert domain_id not in train_data["domain_id"].values
        assert domain_id not in val_data["domain_id"].values
        assert set(test_data["domain_id"].unique()) == {domain_id}


def test_lodo_split_under_3_domains_raises(tmp_path):
    """LODO must raise ValueError if fewer than 3 distinct domains exist."""
    df = _create_mock_index()
    df_2domains = df[df["domain_id"].isin(["domain_a", "domain_b"])]

    with pytest.raises(ValueError) as excinfo:
        generate_lodo_splits(df_2domains, output_dir=tmp_path)
    assert "requires at least 3 distinct domains" in str(excinfo.value)


def test_grouped_split_handles_cross_site_identical_panel_ids():
    """Verify panels with the same local panel_id from different sites do not collide."""
    rows = [
        # Site 1, panel_01
        {"sample_id": "s1_p1_1", "site_id": "site_1", "panel_id": "panel_01", "fault_label": "['Healthy']"},
        {"sample_id": "s1_p1_2", "site_id": "site_1", "panel_id": "panel_01", "fault_label": "['Healthy']"},
        # Site 2, panel_01 (different physical panel at another site)
        {"sample_id": "s2_p1_1", "site_id": "site_2", "panel_id": "panel_01", "fault_label": "['Hotspot']"},
        {"sample_id": "s2_p1_2", "site_id": "site_2", "panel_id": "panel_01", "fault_label": "['Hotspot']"},
    ]
    df = pd.DataFrame(rows)
    from src.splitting.grouped_split import _assign_groups
    groups = _assign_groups(df)
    assert groups[0] == "site_1__panel_01"
    assert groups[1] == "site_1__panel_01"
    assert groups[2] == "site_2__panel_01"
    assert groups[3] == "site_2__panel_01"
    assert groups[0] != groups[2], "Panels from different sites must not share group keys"
