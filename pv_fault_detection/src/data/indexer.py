"""
src/data/indexer.py
Canonical data indexer — runs all registered adapters, builds full.parquet.

Inputs: none (discovers adapters from config registry; actual files under data/raw/)
Outputs:
  - data/processed/index/full.parquet   (one row per canonical sample)
  - data/DATASET_STATUS.md              (PRESENT/MISSING/FAILED per adapter)

CLI: python -m src.data.indexer --config configs/base.yaml [--include-el-substudy]

Spec reference: Section 9.2
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.config import load_config
from src.common.logging_setup import setup_logging
from src.common.paths import (
    data_processed_electrical,
    data_processed_images,
    data_processed_index,
    dataset_status_md,
)
from src.data.adapters.base import CanonicalSample
from src.data.validate import validate_canonical_sample

if TYPE_CHECKING:
    from src.data.adapters.base import DatasetAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

def _build_adapter_registry(active_ids: list[str]) -> dict[str, "DatasetAdapter"]:
    """Import and instantiate only the adapters listed in the config."""
    from src.data.adapters.bimodal_uav import BimodalUAVAdapter
    from src.data.adapters.cali import CaliAdapter
    from src.data.adapters.irsm import IRSMAdapter
    from src.data.adapters.nrel_pvdaq import NRELPVDAQAdapter
    from src.data.adapters.own_site import OwnSiteAdapter
    from src.data.adapters.rgbirpv import RGBIRPVAdapter
    from src.data.adapters.solarfcd import SolarFCDAdapter

    _all_adapters = {
        "irsm": IRSMAdapter(),
        "cali": CaliAdapter(),
        "solarfcd": SolarFCDAdapter(),
        "rgbirpv": RGBIRPVAdapter(),
        "bimodal_uav": BimodalUAVAdapter(),
        "nrel_pvdaq": NRELPVDAQAdapter(),
    }
    # own_site_* adapters are discovered dynamically
    from src.common.paths import data_raw
    raw_root = data_raw()
    for site_dir in raw_root.iterdir():
        if site_dir.is_dir() and site_dir.name.startswith("own_site"):
            _all_adapters[site_dir.name] = OwnSiteAdapter(site_id=site_dir.name)

    return {k: v for k, v in _all_adapters.items() if k in active_ids}


# ---------------------------------------------------------------------------
# Main indexer logic
# ---------------------------------------------------------------------------

def run_indexer(cfg: dict, include_el_substudy: bool = False) -> dict[str, str]:
    """Run all active adapters, write canonical index, update DATASET_STATUS.md.

    Returns:
        dict {dataset_id: status_string}
    """
    active_ids = list(cfg.get("data", {}).get("adapters", []))
    if include_el_substudy:
        from src.data.adapters.elpv import ELPVAdapter
        active_ids.append("elpv")

    adapters = _build_adapter_registry(active_ids)
    if include_el_substudy and "elpv" not in adapters:
        from src.data.adapters.elpv import ELPVAdapter
        adapters["elpv"] = ELPVAdapter()

    all_records: list[dict] = []
    statuses: dict[str, str] = {}

    for adapter_id, adapter in adapters.items():
        logger.info(f"[{adapter_id}] Validating raw data...")
        if not adapter.validate_raw():
            statuses[adapter_id] = "MISSING"
            logger.warning(f"[{adapter_id}] MISSING — skipping (see data/DATASET_STATUS.md)")
            continue

        try:
            count = 0
            error_count = 0
            for raw_sample in adapter.iter_samples():
                try:
                    canonical = adapter.to_canonical(raw_sample)
                    errors = validate_canonical_sample(canonical)
                    if errors:
                        logger.warning(
                            f"[{adapter_id}] Validation errors for {canonical.sample_id}: {errors}"
                        )
                        error_count += 1
                        continue
                    # Copy/process media to data/processed/ and update paths
                    canonical = _process_media(canonical, adapter_id)
                    all_records.append(_canonical_to_dict(canonical))
                    count += 1
                except Exception as e:
                    logger.error(f"[{adapter_id}] Error processing sample {raw_sample.original_id}: {e}")
                    error_count += 1

            if error_count > 0:
                statuses[adapter_id] = f"PARTIAL ({count} OK, {error_count} errors)"
            else:
                statuses[adapter_id] = f"PRESENT ({count} samples)"
            logger.info(f"[{adapter_id}] {statuses[adapter_id]}")

        except Exception as e:
            statuses[adapter_id] = f"FAILED: {e}"
            logger.error(f"[{adapter_id}] Adapter failed: {e}")

    # --- Write Parquet index ---
    _write_parquet(all_records, cfg)

    # --- Update DATASET_STATUS.md ---
    _write_dataset_status_md(statuses, adapters)

    return statuses


def _process_media(sample: CanonicalSample, adapter_id: str) -> CanonicalSample:
    """Ensure media paths on sample are canonical relative paths to existing files."""
    from src.common.paths import PROJECT_ROOT

    # 1. RGB
    if sample.rgb_path:
        src_p = Path(sample.rgb_path)
        if not src_p.is_absolute():
            src_p = PROJECT_ROOT / src_p
        if src_p.exists():
            try:
                sample.rgb_path = str(src_p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                sample.rgb_path = str(src_p).replace("\\", "/")

    # 2. Thermal
    if sample.thermal_path:
        src_p = Path(sample.thermal_path)
        if not src_p.is_absolute():
            src_p = PROJECT_ROOT / src_p
        if src_p.exists():
            try:
                sample.thermal_path = str(src_p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                sample.thermal_path = str(src_p).replace("\\", "/")

    # 3. Electrical
    if sample.electrical_path:
        src_p = Path(sample.electrical_path)
        if not src_p.is_absolute():
            src_p = PROJECT_ROOT / src_p
        if src_p.exists():
            try:
                sample.electrical_path = str(src_p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                sample.electrical_path = str(src_p).replace("\\", "/")

    return sample


def _canonical_to_dict(sample: CanonicalSample) -> dict:
    """Convert CanonicalSample to a flat dict for Parquet serialization."""
    import json
    return {
        "sample_id": sample.sample_id,
        "domain_id": sample.domain_id,
        "site_id": sample.site_id,
        "panel_id": sample.panel_id,
        "inspection_pass_id": sample.inspection_pass_id,
        "timestamp": sample.timestamp,
        "fault_label": json.dumps(sample.fault_label),  # list -> JSON string for Parquet
        "rgb_path": sample.rgb_path,
        "thermal_path": sample.thermal_path,
        "thermal_raw_path": sample.thermal_raw_path,
        "electrical_path": sample.electrical_path,
        "environment": json.dumps(sample.environment) if sample.environment else None,
        "split_hint": sample.split_hint,
        "metadata": json.dumps(sample.metadata),
    }


def _write_parquet(records: list[dict], cfg: dict) -> Path:
    """Write all canonical records to data/processed/index/full.parquet."""
    import pandas as pd
    index_dir = data_processed_index()
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / "full.parquet"

    if not records:
        logger.warning("No records to index. Writing empty Parquet file.")
        df = pd.DataFrame(columns=[
            "sample_id", "domain_id", "site_id", "panel_id", "inspection_pass_id",
            "timestamp", "fault_label", "rgb_path", "thermal_path", "thermal_raw_path",
            "electrical_path", "environment", "split_hint", "metadata",
        ])
    else:
        df = pd.DataFrame(records)

    df.to_parquet(out_path, index=False)
    logger.info(f"Canonical index written: {out_path} ({len(df)} rows)")
    return out_path


def _write_dataset_status_md(statuses: dict[str, str], adapters: dict) -> None:
    """Write data/DATASET_STATUS.md with current status for all adapters."""
    from datetime import timezone
    lines = [
        "# Dataset Status\n\n",
        f"> Auto-generated by `python -m src.data.indexer`. "
        f"Last updated: {datetime.now(timezone.utc).isoformat()}\n\n",
        "| Dataset | Domain ID | Status |\n",
        "|---|---|---|\n",
    ]
    for ds_id, status in sorted(statuses.items()):
        lines.append(f"| {ds_id} | {ds_id} | {status} |\n")

    out_path = dataset_status_md()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Dataset status written to {out_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Parquet index from all adapters.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--include-el-substudy", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)
    statuses = run_indexer(cfg, include_el_substudy=args.include_el_substudy)

    print("\nDataset indexing complete:")
    for ds_id, status in sorted(statuses.items()):
        print(f"  {ds_id:25s} {status}")


if __name__ == "__main__":
    main()
