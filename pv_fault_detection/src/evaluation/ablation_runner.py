import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
import yaml

from src.common.paths import PROJECT_ROOT, results_dir

logger = logging.getLogger(__name__)


def run_ablation_matrix(
    matrix_path: str | Path,
    dry_run: bool = False,
    extra_overrides: list[str] | None = None,
) -> dict:
    """Execute all rows defined in ablation_matrix.yaml."""
    matrix_p = Path(matrix_path)
    if not matrix_p.exists():
        raise FileNotFoundError(f"Ablation matrix not found at {matrix_path}")

    with open(matrix_p) as f:
        matrix_data = yaml.safe_load(f)

    rows = matrix_data.get("rows", [])
    logger.info(f"Loaded {len(rows)} ablation rows from {matrix_path}")

    results_summary = []
    for row in rows:
        row_id = row.get("id")
        name = row.get("name")
        cfg_path = row.get("config")
        protocol = row.get("protocol", "random_split")

        sweeps = row.get("overrides_sweep")
        if sweeps:
            runs_to_exec = sweeps
        else:
            runs_to_exec = [row.get("overrides", [])]

        for sweep_idx, base_ovr in enumerate(runs_to_exec):
            ovr_list = list(base_ovr) if isinstance(base_ovr, list) else [base_ovr]
            if extra_overrides:
                ovr_list.extend(extra_overrides)

            run_tag = f"row_{row_id}_{name}_sweep_{sweep_idx}" if sweeps else f"row_{row_id}_{name}"
            logger.info(f"\n{'='*70}\n[Ablation Row {row_id}] {name} (Protocol: {protocol}) | Overrides: {ovr_list}\n{'='*70}")

            if protocol == "lodo":
                cmd = [sys.executable, "-m", "src.evaluation.lodo_runner", "--config", cfg_path]
                if ovr_list:
                    cmd.extend(["--override"] + ovr_list)
            else:
                cmd = [sys.executable, "-m", "src.experiments.run", "--config", cfg_path]
                if ovr_list:
                    cmd.extend(["--override"] + ovr_list)

            if dry_run:
                print(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
                results_summary.append({
                    "row_id": row_id,
                    "name": name,
                    "run_tag": run_tag,
                    "command": " ".join(cmd),
                    "status": "dry_run",
                })
            else:
                logger.info(f"Executing: {' '.join(cmd)}")
                proc = subprocess.run(cmd, capture_output=True, text=True)
                success = proc.returncode == 0
                if not success:
                    logger.warning(f"Row {row_id} ({name}) failed with code {proc.returncode}:\n{proc.stderr}")
                results_summary.append({
                    "row_id": row_id,
                    "name": name,
                    "run_tag": run_tag,
                    "command": " ".join(cmd),
                    "exit_code": proc.returncode,
                    "success": success,
                })

    summary_file = PROJECT_ROOT / "results" / "ablation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    return {"results": results_summary, "total_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 9-row ablation matrix.")
    parser.add_argument("--matrix", default="src/evaluation/ablation_matrix.yaml", help="Path to ablation_matrix.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--override", nargs="*", default=[], metavar="key=value")
    args = parser.parse_args()

    summary = run_ablation_matrix(
        matrix_path=args.matrix,
        dry_run=args.dry_run,
        extra_overrides=args.override,
    )
    print(f"Ablation execution complete: {len(summary['results'])} experiments processed.")


if __name__ == "__main__":
    main()
