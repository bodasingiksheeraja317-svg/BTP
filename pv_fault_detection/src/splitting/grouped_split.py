"""
src/splitting/grouped_split.py
Group-aware split — ensures no panel_id or inspection_pass_id crosses split boundaries.

Grouping priority:
  1. panel_id (where non-null) — primary leakage-prevention key
  2. inspection_pass_id (where panel_id is null)
  3. sample_id with a loud warning (degraded mode for public datasets with no identifiers)

Spec reference: Sections 9.4, 9.5 (grouping key hierarchy)
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def grouped_split(
    index_df,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split a DataFrame such that no group (panel/pass/sample) crosses boundaries.

    Returns: (train_df, val_df, test_df)
    """
    groups = _assign_groups(index_df)
    unique_groups = list(dict.fromkeys(groups))  # preserve order, deduplicate
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    mask_train = [g in train_groups for g in groups]
    mask_val = [g in val_groups for g in groups]
    mask_test = [g in test_groups for g in groups]

    return (
        index_df[mask_train].reset_index(drop=True),
        index_df[mask_val].reset_index(drop=True),
        index_df[mask_test].reset_index(drop=True),
    )


def _assign_groups(df) -> list[str]:
    """Assign a group key to each row using site_id__panel_id > site_id__inspection_pass_id > sample_id."""
    groups = []
    degraded_count = 0
    for _, row in df.iterrows():
        site_id = str(row.get("site_id") or row.get("domain_id") or "site")
        pid = row.get("panel_id")
        pass_id = row.get("inspection_pass_id")
        if pid and str(pid) != "None" and str(pid) != "nan":
            groups.append(f"{site_id}__{pid}")
        elif pass_id and str(pass_id) != "None" and str(pass_id) != "nan":
            groups.append(f"{site_id}__{pass_id}")
        else:
            groups.append(str(row["sample_id"]))
            degraded_count += 1

    if degraded_count > 0:
        warnings.warn(
            f"{degraded_count}/{len(df)} rows have no panel_id or inspection_pass_id. "
            "Falling back to sample_id grouping — leakage prevention is DEGRADED for these rows.",
            UserWarning,
            stacklevel=3,
        )
    return groups
