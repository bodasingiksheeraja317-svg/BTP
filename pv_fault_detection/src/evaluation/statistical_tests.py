"""
src/evaluation/statistical_tests.py
Bootstrap CIs, Wilcoxon signed-rank tests, effect sizes.

Spec Section 14.3:
  - Bootstrap resampling (1,000 resamples) for 95% CI on every headline metric.
  - Wilcoxon signed-rank test (paired) for two-configuration comparisons.
  - Effect size: rank-biserial correlation alongside every p-value.
  - NEVER a bare point-estimate diff claim in generated tables.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float],
    statistic: Callable = np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: Observed values (e.g. macro-F1 across seeds/rotations).
        statistic: Function to compute on each resample.
        n_resamples: Number of bootstrap resamples (1,000 per spec).
        confidence_level: Confidence level (0.95 per spec).
        seed: Random seed.

    Returns:
        Tuple (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    values = np.array(values)
    point = float(statistic(values))

    boot_stats = [
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_resamples)
    ]
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, ci_lower, ci_upper


def wilcoxon_signed_rank(
    values_a: list[float],
    values_b: list[float],
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test comparing two configurations.

    Args:
        values_a: Metric values for configuration A.
        values_b: Metric values for configuration B.

    Returns:
        Dict with: statistic, p_value, effect_size (rank-biserial correlation).
    """
    from scipy.stats import wilcoxon

    if len(values_a) != len(values_b):
        raise ValueError(
            f"Paired test requires equal-length lists: "
            f"len(a)={len(values_a)}, len(b)={len(values_b)}"
        )
    if len(values_a) < 2:
        logger.warning("Wilcoxon test with < 2 pairs; result unreliable.")
        return {"statistic": None, "p_value": None, "effect_size": None}

    try:
        stat, p_value = wilcoxon(values_a, values_b)
        rbc = _rank_biserial_correlation(values_a, values_b)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "effect_size": float(rbc),
            "n_pairs": len(values_a),
        }
    except Exception as e:
        logger.error(f"Wilcoxon test failed: {e}")
        return {"statistic": None, "p_value": None, "effect_size": None, "error": str(e)}


def _rank_biserial_correlation(
    values_a: list[float], values_b: list[float]
) -> float:
    """Compute rank-biserial correlation as effect size for Wilcoxon test."""
    from scipy.stats import rankdata
    diffs = np.array(values_a) - np.array(values_b)
    n = len(diffs[diffs != 0])
    if n == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs[diffs != 0]))
    r_plus = ranks[diffs[diffs != 0] > 0].sum()
    r_minus = ranks[diffs[diffs != 0] < 0].sum()
    return float((r_plus - r_minus) / (n * (n + 1) / 2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run statistical tests on results.")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    print(f"Statistical tests on {args.results_dir} — full implementation in Stage 13.")


if __name__ == "__main__":
    main()
