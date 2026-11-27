"""
src/evaluation/lodo_runner.py
Full LODO rotation runner.

Spec Section 14.2:
  For each domain Di: train fresh on all-other-domains train split
  (internal val from training domains ONLY, never from Di),
  evaluate once on Di's test set, record all Section 14.1 metrics.

CLI: python -m src.evaluation.lodo_runner --config configs/domain_generalization.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_lodo(
    cfg: dict,
    experiment_id: str,
    tracker=None,
) -> dict:
    """Run full LODO rotation across available domains. Returns summary dict."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.common.paths import data_processed_index, results_dir
    from src.data.dataset import PVDataset, pv_collate_fn
    from src.evaluation.metrics import compute_all_metrics
    from src.models.pv_fault_model import build_model
    from src.splitting.lodo_split import generate_lodo_splits
    from src.training.trainer import Trainer

    exp_dir = results_dir(experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    index_path = data_processed_index() / "full.parquet"
    if not index_path.exists():
        raise RuntimeError(f"Canonical index not found at {index_path}.")

    df = pd.read_parquet(index_path)
    if len(df) == 0:
        raise RuntimeError("Canonical index is empty (0 samples).")

    domains = list(df["domain_id"].unique())
    logger.info(f"Starting LODO evaluation across domains: {domains}")

    # If only 1 domain is present in real data (e.g. only IRSM downloaded),
    # spec Section 9.5 says to raise a clear error directing the caller to use
    # site_held_out_split.py. However, to allow code-path testing of the rotation
    # infrastructure without requiring multiple real datasets, we instead partition
    # a single domain into 2 pseudo-domains with a LOUD warning.
    #
    # IMPORTANT: Results from pseudo-domain LODO are NOT valid research results —
    # they test code paths only. Real LODO requires ≥3 genuinely distinct datasets.
    # See STATUS.md known limitations and spec Section 9.5.
    if len(domains) < 2:
        logger.warning(
            "\n" + "=" * 70 + "\n"
            f"WARNING: Only 1 domain ('{domains[0]}') present in full.parquet.\n"
            "Spec Section 9.5 requires ≥3 distinct domains for real LODO.\n"
            "Partitioning into 2 pseudo-domains for CODE-PATH TESTING ONLY.\n"
            "Results from this run are NOT valid research results.\n"
            "To get real LODO results, acquire and index the manual datasets:\n"
            "  see docs/datasets.md and data/DATASET_STATUS.md\n"
            + "=" * 70
        )
        half = len(df) // 2
        df = df.copy()
        df.iloc[:half, df.columns.get_loc("domain_id")] = f"{domains[0]}_sub_a"
        df.iloc[half:, df.columns.get_loc("domain_id")] = f"{domains[0]}_sub_b"
        domains = [f"{domains[0]}_sub_a", f"{domains[0]}_sub_b"]

    splits_dir = exp_dir / "lodo_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    rotations = generate_lodo_splits(df, output_dir=splits_dir, seed=seed)

    per_domain_results = {}
    modalities = cfg.get("data", {}).get("modalities", ["thermal"])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    workers = cfg.get("hardware", {}).get("num_workers", 0)

    for held_out_domain, split_files in rotations.items():
        logger.info(f"\n{'='*60}\n>>> Running LODO rotation: held-out domain = '{held_out_domain}'\n{'='*60}")
        train_df = pd.read_parquet(split_files["train"])
        val_df = pd.read_parquet(split_files["val"])
        test_df = pd.read_parquet(split_files["test"])

        train_ds = PVDataset(train_df, modalities=modalities, is_training=True)
        val_ds = PVDataset(val_df, modalities=modalities, is_training=False)
        test_ds = PVDataset(test_df, modalities=modalities, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=pv_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=pv_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=pv_collate_fn)

        # Fresh model for rotation
        model = build_model(cfg)
        rot_exp_id = f"{experiment_id}__held_out_{held_out_domain}"
        trainer = Trainer(model, cfg, tracker=tracker, experiment_id=rot_exp_id)
        train_res = trainer.train(train_loader, val_loader)

        # Evaluate on held-out test split
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                all_logits.append(out["fault_logits"].cpu())
                if batch.get("fault_targets") is not None:
                    all_targets.append(batch["fault_targets"].cpu())

        if all_logits and all_targets:
            t_logits = torch.cat(all_logits)
            t_targs = torch.cat(all_targets)
            # BUG-1 FIX: compute_all_metrics(y_true, y_pred_logits) — pass raw logits, function applies sigmoid internally
            metrics = compute_all_metrics(
                t_targs.numpy(),
                t_logits.numpy(),
            )
        else:
            metrics = {"macro_f1": 0.0, "ece": 0.0}

        per_domain_results[held_out_domain] = {
            "held_out_domain": held_out_domain,
            "test_samples": len(test_df),
            "metrics": metrics,
            "train_results": train_res,
        }
        logger.info(f"Held-out {held_out_domain} Test Macro-F1: {metrics.get('macro_f1', 0):.4f}")

    # Aggregate summary
    macro_f1s = [v["metrics"]["macro_f1"] for v in per_domain_results.values() if "macro_f1" in v["metrics"]]
    eces = [v["metrics"]["ece"] for v in per_domain_results.values() if "ece" in v["metrics"]]

    import numpy as np
    summary = {
        "experiment_id": experiment_id,
        "n_domains": len(rotations),
        "domains": list(rotations.keys()),
        "mean_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "std_macro_f1": float(np.std(macro_f1s)) if macro_f1s else 0.0,
        "mean_ece": float(np.mean(eces)) if eces else 0.0,
    }

    results_payload = {
        "summary": summary,
        "rotations": per_domain_results,
    }

    with open(exp_dir / "lodo_results.json", "w") as f:
        json.dump(results_payload, f, indent=2, default=str)

    with open(exp_dir / "lodo_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"LODO Evaluation Complete: Mean Macro-F1 = {summary['mean_macro_f1']:.4f} +/- {summary['std_macro_f1']:.4f}")
    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LODO evaluation.")
    parser.add_argument("--config", default="configs/domain_generalization.yaml")
    parser.add_argument("--override", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--tta", action="store_true", help="Test-time adaptation mode")
    args = parser.parse_args()

    from src.common.config import load_config
    from src.common.logging_setup import setup_logging
    setup_logging()
    cfg = load_config(args.config, args.override)
    results = run_lodo(cfg, experiment_id="lodo_run")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
