"""
src/experiments/run.py
Single experiment entrypoint.

Usage:
  python -m src.experiments.run --config configs/baseline_thermal.yaml
  python -m src.experiments.run --config configs/synthetic_ablation.yaml --override synthetic.ratio=0.25 seed=1

Spec Section 19 (CLI), Section 16 (reproducibility).
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.common.config import load_config, save_config
from src.common.logging_setup import ExperimentTracker, setup_logging
from src.common.paths import checkpoints_dir, results_dir
from src.common.seeding import seed_everything
from src.experiments.registry import generate_experiment_id, register_run
from src.models.pv_fault_model import build_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PV fault detection experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="key=value",
        help="Config overrides in key.subkey=value format"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    setup_logging(log_dir=cfg.get("logging", {}).get("log_dir", "logs/"))

    seed = cfg.get("seed", 42)
    deterministic = cfg.get("training", {}).get("deterministic", True)
    seed_everything(seed, deterministic=deterministic)

    config_name = Path(args.config).stem
    experiment_id = generate_experiment_id(config_name, seed)
    logger.info(f"Experiment ID: {experiment_id}")

    exp_results_dir = results_dir(experiment_id)
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, exp_results_dir / "config.yaml")
    _snapshot_environment(exp_results_dir)
    register_run(experiment_id, cfg, config_path=args.config)

    tracker = ExperimentTracker.from_config(cfg)
    tracker.start_run(experiment_id, cfg)

    try:
        model = build_model(cfg)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        train_loader, val_loader, test_loader = _build_dataloaders(cfg)

        from src.training.trainer import Trainer
        trainer = Trainer(model, cfg, tracker=tracker, experiment_id=experiment_id)
        train_results = trainer.train(train_loader, val_loader)

        logger.info(f"Training complete. Best val macro-F1: {train_results.get('best_val_macro_f1', 0):.4f}")

        # Evaluate on test loader
        import time
        import torch
        from src.evaluation.metrics import compute_all_metrics
        model.eval()
        all_logits, all_targets = [], []
        device = trainer.device
        # Time inference for latency metric (spec Section 14.1)
        latency_start = time.perf_counter()
        n_inferred = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                all_logits.append(out["fault_logits"].cpu())
                if batch.get("fault_targets") is not None:
                    all_targets.append(batch["fault_targets"].cpu())
                n_inferred += out["fault_logits"].shape[0]
        latency_ms_per_sample = (
            (time.perf_counter() - latency_start) * 1000.0 / max(n_inferred, 1)
        )
        model_param_count = sum(p.numel() for p in model.parameters())

        test_metrics = {}
        if all_logits and all_targets:
            test_logits = torch.cat(all_logits)
            test_targs = torch.cat(all_targets)
            # BUG-1 FIX: compute_all_metrics(y_true, y_pred_logits) — pass raw logits, not pre-softmaxed probs
            # (the function applies sigmoid internally)
            test_metrics = compute_all_metrics(
                test_targs.numpy(),
                test_logits.numpy(),
            )
            test_metrics["inference_latency_ms"] = float(latency_ms_per_sample)
            test_metrics["model_param_count"] = int(model_param_count)
            logger.info(
                f"Test Evaluation: Macro-F1 = {test_metrics.get('macro_f1', 0):.4f}, "
                f"ECE = {test_metrics.get('ece', 0):.4f}, "
                f"Latency = {latency_ms_per_sample:.2f} ms/sample"
            )

        ckpt_dir = checkpoints_dir(experiment_id)
        (exp_results_dir / "checkpoint_pointer.txt").write_text(str(ckpt_dir))

        metrics = {
            "experiment_id": experiment_id,
            "train_results": train_results,
            "test_metrics": test_metrics,
            **test_metrics,
        }
        with open(exp_results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    finally:
        tracker.end_run()


def _build_dataloaders(cfg: dict):
    """Build real train/val/test DataLoaders from canonical index."""
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.common.paths import data_processed_index
    from src.data.dataset import PVDataset, pv_collate_fn
    from src.splitting.grouped_split import grouped_split
    from src.synthetic.mixing import SyntheticMixer

    index_path = data_processed_index() / "full.parquet"
    if not index_path.exists():
        raise RuntimeError(f"Canonical index not found at {index_path}. Run python -m src.data.indexer first.")

    df = pd.read_parquet(index_path)
    if len(df) == 0:
        raise RuntimeError("Canonical index is empty (0 samples). Please acquire or index data.")

    # Filter active modalities if needed
    modalities = cfg.get("data", {}).get("modalities", ["thermal"])
    seed = cfg.get("seed", 42)

    # 1. Create splits
    train_df, val_df, test_df = grouped_split(
        df,
        train_frac=cfg.get("splitting", {}).get("train_frac", 0.70),
        val_frac=cfg.get("splitting", {}).get("val_frac", 0.15),
        seed=seed,
    )

    # 2. Synthetic data mixing on train split only
    synth_ratio = cfg.get("synthetic", {}).get("ratio", 0.0)
    if synth_ratio and synth_ratio != 0.0:
        mixer = SyntheticMixer(ratio=synth_ratio, split="train")
        # If electrical synthetic data exists, create synthetic DataFrame records
        from src.common.paths import data_synthetic_electrical
        synth_files = list(data_synthetic_electrical().rglob("*.npz"))
        if synth_files:
            from src.common.paths import PROJECT_ROOT
            synth_records = []
            for sf in synth_files:
                fault_cls = sf.parent.name.replace("_", "/")
                # ISSUE-7 FIX: paths stored in index must be relative to PROJECT_ROOT
                # (PVDataset resolves them as PROJECT_ROOT / path)
                try:
                    rel_path = str(sf.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except ValueError:
                    rel_path = str(sf).replace("\\", "/")
                synth_records.append({
                    "sample_id": f"synth_elec__{sf.stem}",
                    "domain_id": "synth_elec",
                    "site_id": "synth_elec",
                    "fault_label": f"['{fault_cls}']",
                    "rgb_path": None,
                    "thermal_path": None,
                    "electrical_path": rel_path,
                    "environment": None,
                })
            synth_df = pd.DataFrame(synth_records)
            train_df = mixer.mix(train_df, synth_df)

    # 3. Batch size and workers
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    workers = cfg.get("hardware", {}).get("num_workers", 0)

    train_ds = PVDataset(train_df, modalities=modalities, is_training=True)
    val_ds = PVDataset(val_df, modalities=modalities, is_training=False)
    test_ds = PVDataset(test_df, modalities=modalities, is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=pv_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=pv_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, collate_fn=pv_collate_fn
    )

    logger.info(f"DataLoaders built: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} samples.")
    return train_loader, val_loader, test_loader


def _snapshot_environment(results_path: Path) -> None:
    """Snapshot pip freeze to results/<experiment_id>/environment.txt."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        (results_path / "environment.txt").write_text(result.stdout)
    except Exception as e:
        logger.warning(f"Failed to snapshot pip freeze: {e}")


if __name__ == "__main__":
    main()
