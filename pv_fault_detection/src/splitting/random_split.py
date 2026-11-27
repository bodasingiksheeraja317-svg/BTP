"""
src/splitting/random_split.py
Random split utility — for DEBUGGING ONLY.

WARNING: This splitter ignores panel_id/site_id grouping.
NEVER use for reporting headline results. Not safe for research conclusions.
Only for quick debugging and sanity checks.

Spec reference: Section 9.4
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

_UNSAFE_WARNING = (
    "UNSAFE: random_split ignores panel_id/site_id grouping and MUST NOT be used "
    "for headline results. Use grouped_split, site_held_out_split, or lodo_split instead."
)


def random_split(
    index_df,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split a DataFrame randomly into train/val/test.

    Returns: (train_df, val_df, test_df)
    """
    warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=2)
    logger.warning(_UNSAFE_WARNING)

    rng = np.random.default_rng(seed)
    n = len(index_df)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        index_df.iloc[train_idx].reset_index(drop=True),
        index_df.iloc[val_idx].reset_index(drop=True),
        index_df.iloc[test_idx].reset_index(drop=True),
    )
