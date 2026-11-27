"""
src/data/download.py
Dataset acquisition script.

AUTOMATIC datasets: attempts actual download.
MANUAL datasets: prints instructions and expected paths, exits WITHOUT error.
This is correct behavior, not failure.

Hard rule (spec Section 20): never fabricate or guess a URL for a MANUAL dataset.
All failures recorded in data/DATASET_STATUS.md.

CLI: python -m src.data.download --config configs/base.yaml [--include-el-substudy]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

from src.common.config import load_config
from src.common.logging_setup import setup_logging
from src.common.paths import data_raw, dataset_status_md

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

class DatasetInfo(NamedTuple):
    dataset_id: str
    access_mode: str          # "AUTOMATIC" | "MANUAL"
    license_note: str
    download_url: str | None  # None for MANUAL
    manual_instructions: str | None


DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        dataset_id="irsm",
        access_mode="AUTOMATIC",
        license_note="Free for research — verify current terms at Raptor Maps GitHub",
        # TO VERIFY: confirm this URL is still live at build time
        download_url="https://github.com/RaptorMaps/InfraredSolarModules/archive/refs/heads/master.zip",
        manual_instructions=None,
    ),
    DatasetInfo(
        dataset_id="elpv",
        access_mode="AUTOMATIC",
        license_note="Open dataset — cite Deitsch et al. (2019)",
        download_url="https://github.com/zae-bayern/elpv-dataset/archive/refs/heads/master.zip",
        manual_instructions=None,
    ),
    DatasetInfo(
        dataset_id="cali",
        access_mode="MANUAL",
        license_note="Academic use — cite Alfaro-Mejia et al. TO VERIFY exact license.",
        download_url=None,
        manual_instructions=(
            "MANUAL DOWNLOAD REQUIRED: Alfaro-Mejia Cali Thermal Dataset\n"
            "  1. Locate the dataset from the original paper:\n"
            "       Alfaro-Mejia et al. — check arXiv or journal supplementary material.\n"
            "  2. Verify license terms before using in publications.\n"
            "  3. Place extracted files under: data/raw/cali/\n"
            "     Expected structure: data/raw/cali/<fault_class>/<image_file>\n"
            "  4. Re-run: python -m src.data.download --config configs/base.yaml\n"
        ),
    ),
    DatasetInfo(
        dataset_id="solarfcd",
        access_mode="MANUAL",
        license_note="Open — TO VERIFY exact terms at dataset repository.",
        download_url=None,
        manual_instructions=(
            "MANUAL DOWNLOAD REQUIRED: SolarFCD Dataset\n"
            "  1. Find the dataset at: check arXiv/dataset-repo release (TO VERIFY current hosting).\n"
            "  2. Verify license at the dataset repository.\n"
            "  3. Place extracted files under: data/raw/solarfcd/\n"
            "  4. Re-run: python -m src.data.download --config configs/base.yaml\n"
        ),
    ),
    DatasetInfo(
        dataset_id="rgbirpv",
        access_mode="MANUAL",
        license_note="TO VERIFY — check journal supplementary terms.",
        download_url=None,
        manual_instructions=(
            "MANUAL DOWNLOAD REQUIRED: RGBIRPV Dataset (RGB + IR paired)\n"
            "  1. Find the dataset from the paper's journal supplementary / GitHub.\n"
            "     TO VERIFY: check current hosting URL.\n"
            "  2. Verify license before use.\n"
            "  3. Place extracted files under: data/raw/rgbirpv/\n"
            "  4. Re-run: python -m src.data.download --config configs/base.yaml\n"
        ),
    ),
    DatasetInfo(
        dataset_id="bimodal_uav",
        access_mode="MANUAL",
        license_note="TO VERIFY — check paper supplementary terms.",
        download_url=None,
        manual_instructions=(
            "MANUAL DOWNLOAD REQUIRED: Bimodal UAV Dataset (RGB + Thermal)\n"
            "  1. Find the dataset from the paper's supplementary material.\n"
            "     TO VERIFY: current hosting URL.\n"
            "  2. Verify license before use.\n"
            "  3. Place extracted files under: data/raw/bimodal_uav/\n"
            "  4. Re-run: python -m src.data.download --config configs/base.yaml\n"
        ),
    ),
    DatasetInfo(
        dataset_id="nrel_pvdaq",
        access_mode="MANUAL",
        license_note="Public with registration — NREL PVDAQ / Sandia PV fleet.",
        download_url=None,
        manual_instructions=(
            "MANUAL DOWNLOAD REQUIRED: NREL PVDAQ\n"
            "  REQUIRES USER INPUT: NREL account + API key\n"
            "  1. Register at: https://developer.nrel.gov/signup/\n"
            "  2. Add your API key to .env: NREL_API_KEY=<your_key>\n"
            "  3. Export data from the PVDAQ portal.\n"
            "  4. Place exported files under: data/raw/nrel_pvdaq/\n"
            "     Expected structure: data/raw/nrel_pvdaq/<site_id>/<date>.csv\n"
            "  5. Re-run: python -m src.data.download --config configs/base.yaml\n"
        ),
    ),
]

MANUAL_INSTRUCTIONS_SEPARATOR = "\n" + "=" * 70 + "\n"


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_automatic(info: DatasetInfo) -> bool:
    """Attempt to download an AUTOMATIC dataset. Returns True on success."""
    raw_dir = data_raw(info.dataset_id)
    if raw_dir.exists() and any(raw_dir.iterdir()):
        logger.info(f"[{info.dataset_id}] Already present at {raw_dir}. Skipping download.")
        return True

    raw_dir.mkdir(parents=True, exist_ok=True)
    url = info.download_url
    zip_path = raw_dir / "download.zip"

    logger.info(f"[{info.dataset_id}] Downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"[{info.dataset_id}] Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()  # remove zip after extraction
        logger.info(f"[{info.dataset_id}] Extracted to {raw_dir}")
        return True
    except Exception as e:
        logger.error(
            f"[{info.dataset_id}] DOWNLOAD FAILED: {e}\n"
            f"  URL: {url}\n"
            f"  TO VERIFY: Check if this URL is still valid. Record failure in data/DATASET_STATUS.md."
        )
        # Clean up partial download
        if zip_path.exists():
            zip_path.unlink()
        return False


def print_manual_instructions(info: DatasetInfo) -> None:
    """Print manual download instructions for a MANUAL dataset. Never errors."""
    print(MANUAL_INSTRUCTIONS_SEPARATOR)
    print(info.manual_instructions)


def run_download(cfg: dict, include_el_substudy: bool = False) -> dict[str, str]:
    """Run all downloads. Returns status dict {dataset_id: status_string}."""
    active_adapter_ids = set(cfg.get("data", {}).get("adapters", []))
    if include_el_substudy:
        active_adapter_ids.add("elpv")

    statuses: dict[str, str] = {}

    for info in DATASETS:
        if info.dataset_id not in active_adapter_ids and info.dataset_id != "elpv":
            statuses[info.dataset_id] = "SKIPPED (not in active adapters config)"
            continue

        raw_dir = data_raw(info.dataset_id)
        if raw_dir.exists() and any(raw_dir.iterdir()):
            statuses[info.dataset_id] = "PRESENT"
            logger.info(f"[{info.dataset_id}] PRESENT at {raw_dir}")
            continue

        if info.access_mode == "AUTOMATIC":
            success = download_automatic(info)
            statuses[info.dataset_id] = "PRESENT" if success else "DOWNLOAD_FAILED"
        else:  # MANUAL
            print_manual_instructions(info)
            statuses[info.dataset_id] = "MANUAL_PENDING"

    _write_status_md(statuses)
    return statuses


def _write_status_md(statuses: dict[str, str]) -> None:
    """Write/update data/DATASET_STATUS.md with current status."""
    from datetime import datetime, timezone
    lines = [
        "# Dataset Status\n",
        f"> Auto-generated by `python -m src.data.download`. Last updated: {datetime.now(timezone.utc).isoformat()}\n\n",
        "| Dataset | Domain ID | Access Mode | Status | License |\n",
        "|---|---|---|---|---|\n",
    ]
    info_by_id = {d.dataset_id: d for d in DATASETS}
    for dataset_id, status in sorted(statuses.items()):
        info = info_by_id.get(dataset_id)
        access = info.access_mode if info else "UNKNOWN"
        license_note = info.license_note if info else ""
        lines.append(f"| {dataset_id} | {dataset_id} | {access} | {status} | {license_note} |\n")

    out_path = dataset_status_md()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Dataset status written to {out_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AUTOMATIC datasets; print instructions for MANUAL ones."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="key=value",
        help="Config overrides in key.subkey=value format"
    )
    parser.add_argument(
        "--include-el-substudy", action="store_true",
        help="Also download ELPV (optional EL sub-study, excluded from default registry)"
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, args.override)
    statuses = run_download(cfg, include_el_substudy=args.include_el_substudy)

    print("\n" + "=" * 70)
    print("DATASET STATUS SUMMARY")
    print("=" * 70)
    for ds_id, status in sorted(statuses.items()):
        print(f"  {ds_id:25s} {status}")
    print(f"\nFull status: {dataset_status_md()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
