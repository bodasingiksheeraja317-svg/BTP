"""
src/splitting/site_held_out_split.py
Single-site held-out split for cross-domain evaluation.

Holds out one complete domain (site x camera x climate combination) as test set.
The remaining domains form the train/val pool.

Spec reference: Section 9.5
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def site_held_out_split(
    index_df,
    held_out_domain: str,
    val_frac_within_source: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Create a cross-domain split by holding out one complete domain.

    Args:
        index_df: Canonical Parquet DataFrame.
        held_out_domain: The domain_id to hold out entirely as test set.
        val_frac_within_source: Fraction of source-domain data for validation.
        seed: Random seed.

    Returns:
        (train_df, val_df, test_df) where test_df contains ONLY held_out_domain.
    """
    test_df = index_df[index_df["domain_id"] == held_out_domain].reset_index(drop=True)
    source_df = index_df[index_df["domain_id"] != held_out_domain].reset_index(drop=True)

    if len(test_df) == 0:
        raise ValueError(
            f"held_out_domain='{held_out_domain}' has no rows in the index. "
            f"Available domains: {sorted(index_df['domain_id'].unique())}"
        )

    # Split source domains into train/val using grouped split
    from src.splitting.grouped_split import grouped_split
    train_df, val_df, _ = grouped_split(
        source_df,
        train_frac=1.0 - val_frac_within_source,
        val_frac=val_frac_within_source,
        seed=seed,
    )

    logger.info(
        f"Site-held-out split: held_out='{held_out_domain}' "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df
