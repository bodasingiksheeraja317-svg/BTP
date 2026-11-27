"""
tests/leakage/test_no_site_in_train_and_lodo_test.py
Leakage test: held-out domain_id has ZERO rows in that LODO rotation's train/val partitions.

This is a LOAD-BEARING test. If it fails, the pipeline splitting code is wrong, not this test.
"""
import pandas as pd
import pytest
from src.splitting.lodo_split import generate_lodo_splits

pytestmark = [pytest.mark.leakage]


def test_lodo_zero_site_leakage(tmp_path):
    """Verify that in every LODO rotation, the held-out site is absent from training & validation."""
    records = []
    domains = ["irsm_site1", "cali_site2", "solarfcd_site3", "own_site_a"]
    for d in domains:
        for idx in range(20):
            records.append({
                "sample_id": f"{d}__{idx}",
                "domain_id": d,
                "site_id": d,
                "panel_id": f"panel_{idx}",
                "inspection_pass_id": "pass_1",
                "fault_label": "['Healthy']",
            })
    df = pd.DataFrame(records)

    rotations = generate_lodo_splits(df, output_dir=tmp_path, seed=42)

    for held_out_domain, files in rotations.items():
        train_df = pd.read_parquet(files["train"])
        val_df = pd.read_parquet(files["val"])
        test_df = pd.read_parquet(files["test"])

        # 1. Held-out domain has ZERO rows in train and val
        train_leakage = train_df[train_df["domain_id"] == held_out_domain]
        val_leakage = val_df[val_df["domain_id"] == held_out_domain]

        assert len(train_leakage) == 0, (
            f"DATA LEAKAGE in rotation '{held_out_domain}': "
            f"{len(train_leakage)} rows found in train split!"
        )
        assert len(val_leakage) == 0, (
            f"DATA LEAKAGE in rotation '{held_out_domain}': "
            f"{len(val_leakage)} rows found in val split!"
        )

        # 2. Test split contains ONLY the held-out domain
        non_test_domains = test_df[test_df["domain_id"] != held_out_domain]
        assert len(non_test_domains) == 0, (
            f"Contamination in rotation '{held_out_domain}' test set: "
            f"unexpected domains {non_test_domains['domain_id'].unique()}"
        )

        # 3. All samples of held_out_domain are in test_df
        total_held_out_samples = len(df[df["domain_id"] == held_out_domain])
        assert len(test_df) == total_held_out_samples
