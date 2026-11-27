"""
src/splitting/lodo_split.py
Leave-One-Domain-Out (LODO) split generator.

For each domain Di, produces:
  - lodo_<Di>_train.parquet: all other domains (train portion)
  - lodo_<Di>_val.parquet: internal val from other domains only
  - lodo_<Di>_test.parquet: ONLY Di, held out entirely

Critical: Di has ZERO exposure during that rotation's training/val/HP tuning.
This is enforced by construction: test is taken directly from the Di rows.

Fallback rule: if < 3 distinct domains exist, raises ValueError directing to
site_held_out_split. LODO with <3 domains is not silently supported.

Spec reference: Section 9.5, Section 14.2

CLI: python -m src.splitting.lodo_split --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_DOMAINS_FOR_LODO = 3


def generate_lodo_splits(
    index_df,
    output_dir: str | Path,
    val_frac_within_source: float = 0.15,
    seed: int = 42,
) -> dict[str, dict]:
    """Generate all LODO rotation index files.

    Args:
        index_df: Canonical Parquet index DataFrame.
        output_dir: Directory to write lodo_*.parquet files.
        val_frac_within_source: Val fraction from source-domain data per rotation.
        seed: Random seed.

    Returns:
        Dict {domain_id: {"train": Path, "val": Path, "test": Path}}

    Raises:
        ValueError: If fewer than MIN_DOMAINS_FOR_LODO distinct domains exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = sorted(index_df["domain_id"].unique().tolist())
    if len(domains) < MIN_DOMAINS_FOR_LODO:
        raise ValueError(
            f"LODO requires at least {MIN_DOMAINS_FOR_LODO} distinct domains, "
            f"but only {len(domains)} found: {domains}. "
            "Use site_held_out_split.py with domain-stratified k-fold instead "
            "(set evaluation.protocol=site_held_out in your config)."
        )

    rotation_paths: dict[str, dict] = {}
    for held_out_domain in domains:
        logger.info(f"Generating LODO rotation: held_out='{held_out_domain}'")
        from src.splitting.site_held_out_split import site_held_out_split
        train_df, val_df, test_df = site_held_out_split(
            index_df,
            held_out_domain=held_out_domain,
            val_frac_within_source=val_frac_within_source,
            seed=seed,
        )

        assert len(train_df[train_df["domain_id"] == held_out_domain]) == 0, (
            f"LEAKAGE: {held_out_domain} found in train split!"
        )
        assert len(val_df[val_df["domain_id"] == held_out_domain]) == 0, (
            f"LEAKAGE: {held_out_domain} found in val split!"
        )

        train_path = output_dir / f"lodo_{held_out_domain}_train.parquet"
        val_path = output_dir / f"lodo_{held_out_domain}_val.parquet"
        test_path = output_dir / f"lodo_{held_out_domain}_test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        rotation_paths[held_out_domain] = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }
        logger.info(
            f"  [{held_out_domain}] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    logger.info(f"LODO splits written to {output_dir} ({len(domains)} rotations)")
    return rotation_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LODO split index files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[], metavar="key=value")
    args = parser.parse_args()

    from src.common.config import load_config
    from src.common.logging_setup import setup_logging
    from src.common.paths import data_processed_index
    setup_logging()
    cfg = load_config(args.config, args.override)

    import pandas as pd
    index_path = cfg["data"]["index_path"]
    df = pd.read_parquet(index_path)
    output_dir = data_processed_index()
    generate_lodo_splits(df, output_dir=output_dir, seed=cfg.get("seed", 42))
    print(f"LODO splits written to {output_dir}")


if __name__ == "__main__":
    main()
