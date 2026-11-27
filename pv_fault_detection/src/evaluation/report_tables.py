"""
src/evaluation/report_tables.py
Auto-generate the 6 research tables (spec Section 25) from real metrics.json files.

IMPORTANT (spec Section 2 + STATUS.md BUG-2 fix):
  - Every number must trace to a real metrics.json from a real run.
  - Rows for experiments that have NOT been run yet must show NOT_YET_RUN.
  - NEVER hardcode or fabricate plausible-looking result values.

Tables generated:
  1. Single-modality performance (ablation rows 1-3)
  2. Multimodal ablation (ablation rows 4-6)
  3. Fusion mechanism comparison (row 6)
  4. Synthetic-data ratio sweep (row 7)
  5. Cross-site generalization gap (row 9)
  6. DG method comparison (row 8)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NOT_RUN = "NOT_YET_RUN"


def _load_all_runs(results_dir: Path) -> list[dict]:
    """Load all metrics.json files from results/ subfolders. Returns list of run dicts."""
    runs = []
    for mf in results_dir.rglob("metrics.json"):
        try:
            with open(mf) as f:
                data = json.load(f)
            data["_metrics_file"] = str(mf)
            data["_experiment_dir"] = str(mf.parent.name)
            runs.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {mf}: {e}")
    logger.info(f"Loaded {len(runs)} real experiment runs from {results_dir}")
    return runs


def _find_run(runs: list[dict], config_tag: str) -> dict | None:
    """Find the most recent real run matching a config name tag (partial match on experiment_id)."""
    candidates = [r for r in runs if config_tag in r.get("experiment_id", r.get("_experiment_dir", ""))]
    if not candidates:
        return None
    # Return most recent (sort by experiment_id which has timestamp)
    candidates.sort(key=lambda r: r.get("experiment_id", r.get("_experiment_dir", "")), reverse=True)
    return candidates[0]


def _get_metric(run: dict | None, key: str, nested_key: str = "test_metrics") -> str:
    """Safely extract a metric from a run dict. Returns NOT_RUN if missing."""
    if run is None:
        return NOT_RUN
    # Try top-level first, then nested under test_metrics
    val = run.get(key)
    if val is None:
        val = run.get(nested_key, {}).get(key)
    if val is None:
        return NOT_RUN
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _get_lodo_summary(results_dir: Path, config_tag: str) -> dict | None:
    """Find a lodo_summary.json for a given experiment config tag."""
    for lsf in results_dir.rglob("lodo_summary.json"):
        if config_tag in str(lsf.parent):
            try:
                with open(lsf) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def generate_report_tables(results_dir_path: str | Path, out_dir_path: str | Path) -> dict[str, str]:
    """Generate all 6 research tables from real metrics.json files.

    Any experiment that has not been run shows NOT_YET_RUN in all metric cells.
    NO fabricated numbers are ever written.
    """
    results_dir = Path(results_dir_path)
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_all_runs(results_dir)
    tables: dict[str, str] = {}

    # ---- Table 1: Single-Modality Performance (ablation rows 1-3) ----
    t1_rows = []
    for tag, label in [
        ("baseline_rgb", "RGB-only (Baseline)"),
        ("baseline_thermal", "Thermal-only (Baseline)"),
        ("baseline_electrical", "Electrical-only (Baseline)"),
    ]:
        run = _find_run(runs, tag)
        t1_rows.append({
            "Experiment": label,
            "Macro-F1": _get_metric(run, "macro_f1"),
            "Balanced Accuracy": _get_metric(run, "balanced_accuracy"),
            "ECE": _get_metric(run, "ece"),
            "Hotspot Recall": _get_metric(run, "hotspot_recall"),
            "Params": _get_metric(run, "model_param_count"),
            "Latency (ms/sample)": _get_metric(run, "inference_latency_ms"),
            "Source": run["_experiment_dir"] if run else NOT_RUN,
        })
    tables["table1_single_modality"] = pd.DataFrame(t1_rows).to_markdown(index=False)

    # ---- Table 2: Multimodal Ablation (ablation rows 4-6) ----
    t2_rows = []
    for tag, label in [
        ("multimodal_rgb_thermal", "RGB + Thermal (Bimodal)"),
        ("multimodal_full", "Full Trimodal (RGB+Thermal+Elec)"),
        ("fusion_late_vs_crossattn", "Late Fusion vs Cross-Attention"),
    ]:
        run = _find_run(runs, tag)
        t2_rows.append({
            "Experiment": label,
            "Macro-F1": _get_metric(run, "macro_f1"),
            "ECE": _get_metric(run, "ece"),
            "Source": run["_experiment_dir"] if run else NOT_RUN,
        })
    tables["table2_multimodal_ablation"] = pd.DataFrame(t2_rows).to_markdown(index=False)

    # ---- Table 3: Synthetic Ratio Sweep (ablation row 7) ----
    t3_rows = []
    for ratio in ["0.0", "0.1", "0.25", "0.5", "1.0", "unlimited", "pretrain_finetune", "domain_randomized"]:
        run = _find_run(runs, f"synthetic_ablation")
        # Sub-filter by ratio override if multiple runs exist
        if run and ratio not in run.get("experiment_id", ""):
            # Check all runs for this specific ratio
            ratio_runs = [r for r in runs if "synthetic_ablation" in r.get("experiment_id", "")
                          and f"ratio={ratio}" in r.get("experiment_id", "")]
            run = ratio_runs[-1] if ratio_runs else None
        t3_rows.append({
            "Synthetic Ratio": ratio,
            "Macro-F1": _get_metric(run, "macro_f1"),
            "ECE": _get_metric(run, "ece"),
            "Source": run["_experiment_dir"] if run else NOT_RUN,
        })
    tables["table3_synthetic_sweep"] = pd.DataFrame(t3_rows).to_markdown(index=False)

    # ---- Table 4: Synthetic Data Impact on In-Domain vs Cross-Site (from row 7 + row 9) ----
    t4_rows = []
    for protocol, tag, label in [
        ("random_split", "multimodal_full", "In-Domain (Grouped Split)"),
        ("lodo", "lodo_final", "Cross-Site LODO (Zero-Shot)"),
    ]:
        if protocol == "lodo":
            lodo_summary = _get_lodo_summary(results_dir, tag)
            if lodo_summary:
                mean_f1 = lodo_summary.get("mean_macro_f1", None)
                t4_rows.append({
                    "Evaluation Protocol": label,
                    "Macro-F1": f"{mean_f1:.4f}" if mean_f1 is not None else NOT_RUN,
                    "Std-F1": f"{lodo_summary.get('std_macro_f1', 0):.4f}",
                    "Source": "lodo_summary.json",
                })
            else:
                t4_rows.append({
                    "Evaluation Protocol": label,
                    "Macro-F1": NOT_RUN,
                    "Std-F1": NOT_RUN,
                    "Source": NOT_RUN,
                })
        else:
            run = _find_run(runs, tag)
            t4_rows.append({
                "Evaluation Protocol": label,
                "Macro-F1": _get_metric(run, "macro_f1"),
                "Std-F1": "N/A (single run)",
                "Source": run["_experiment_dir"] if run else NOT_RUN,
            })
    tables["table4_generalization_gap"] = pd.DataFrame(t4_rows).to_markdown(index=False)

    # ---- Table 5: DG Method Comparison (ablation row 8) ----
    t5_rows = []
    for dg_method in ["none", "coral", "adversarial", "domain_norm"]:
        lodo_summary = _get_lodo_summary(results_dir, f"domain_generalization")
        # Try to find DG-specific lodo summary
        dg_runs = [r for r in runs if "domain_generalization" in r.get("experiment_id", "")
                   and f"dg_method={dg_method}" in r.get("experiment_id", "")]
        dg_run = dg_runs[-1] if dg_runs else None
        t5_rows.append({
            "DG Method": dg_method,
            "LODO Mean Macro-F1": _get_metric(dg_run, "macro_f1") if dg_run else NOT_RUN,
            "LODO Std Macro-F1": NOT_RUN,
            "Source": dg_run["_experiment_dir"] if dg_run else NOT_RUN,
        })
    tables["table5_dg_methods"] = pd.DataFrame(t5_rows).to_markdown(index=False)

    # ---- Table 6: Per-Class Breakdown (from best completed run) ----
    classes = [
        "Healthy", "Hotspot", "Open/Short-circuit", "Shading/Obstruction",
        "Soiling", "Structural (visual)", "Electrical/Diode fault",
    ]
    # Find the run with the highest macro_f1 (most meaningful to show per-class)
    best_run = None
    best_f1 = -1.0
    for r in runs:
        f1 = r.get("macro_f1", r.get("test_metrics", {}).get("macro_f1", 0.0) or 0.0)
        if isinstance(f1, (int, float)) and f1 > best_f1:
            best_f1 = float(f1)
            best_run = r

    t6_rows = []
    for cls in classes:
        if best_run and "per_class" in best_run:
            cls_data = best_run["per_class"].get(cls, {})
            t6_rows.append({
                "Fault Class": cls,
                "Precision": f"{cls_data.get('precision', 0):.4f}" if cls_data else NOT_RUN,
                "Recall": f"{cls_data.get('recall', 0):.4f}" if cls_data else NOT_RUN,
                "F1-Score": f"{cls_data.get('f1', 0):.4f}" if cls_data else NOT_RUN,
                "Source": best_run["_experiment_dir"] if best_run else NOT_RUN,
            })
        else:
            t6_rows.append({
                "Fault Class": cls,
                "Precision": NOT_RUN,
                "Recall": NOT_RUN,
                "F1-Score": NOT_RUN,
                "Source": NOT_RUN,
            })
    tables["table6_per_class_breakdown"] = pd.DataFrame(t6_rows).to_markdown(index=False)

    # Write combined markdown report
    full_report_path = out_dir / "research_tables.md"
    with open(full_report_path, "w") as f:
        f.write("# PV Fault Detection Research Report — Auto-Generated Tables\n\n")
        f.write(f"> Generated from {len(runs)} completed experiment runs in `{results_dir}`.\n")
        f.write("> Rows marked `NOT_YET_RUN` require the corresponding experiment to be executed first.\n")
        f.write("> **No values in this report are fabricated or hardcoded.**\n\n")
        for name, md_content in tables.items():
            f.write(f"## {name.replace('_', ' ').title()}\n\n")
            f.write(md_content + "\n\n")

    logger.info(f"Research tables written to {full_report_path} (from {len(runs)} real runs)")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research tables from real results.")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--out", default="reports/")
    args = parser.parse_args()

    tables = generate_report_tables(args.results_dir, args.out)
    print(f"Generated {len(tables)} research tables in {args.out}.")
    print("Note: Rows marked NOT_YET_RUN require running the corresponding experiment first.")


if __name__ == "__main__":
    main()
