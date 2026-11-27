"""
src/data/adapters/nrel_pvdaq.py
Adapter for NREL PVDAQ / Sandia PV fleet data.

Dataset: NREL PVDAQ
Domain ID: nrel_pvdaq
Modality: Electrical + environmental (real, healthy-dominant)
Role: Healthy-domain diversity + irradiance/temp normalization source
Access: MANUAL (registration required — NREL account + API key needed)
License: Public with registration. REQUIRES USER INPUT: NREL_API_KEY.

REQUIRES USER INPUT:
  1. Register at https://developer.nrel.gov/signup/
  2. Place your API key in .env as NREL_API_KEY=...
  3. Export data and place under data/raw/nrel_pvdaq/
  See docs/datasets.md for expected directory structure.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from src.common.paths import data_raw
from src.data.adapters.base import CanonicalSample, DatasetAdapter, RawSample

logger = logging.getLogger(__name__)

DATASET_ID = "nrel_pvdaq"
DOMAIN_ID = "nrel_pvdaq"
LICENSE = "Public data with registration — NREL PVDAQ / Sandia PV fleet"


class NRELPVDAQAdapter(DatasetAdapter):
    """NREL PVDAQ electrical+environmental adapter (MANUAL download required)."""

    def __init__(self, raw_root: Path | None = None, label_map: dict | None = None):
        self._raw_root = raw_root or data_raw(DATASET_ID)
        self._label_map = label_map or _load_label_map()

    def dataset_id(self) -> str:
        return DATASET_ID

    def license_tag(self) -> str:
        return LICENSE

    def validate_raw(self) -> bool:
        root = self._raw_root
        if not root.exists():
            logger.warning(
                f"NREL PVDAQ raw directory not found: {root}.\n"
                "MANUAL acquisition required:\n"
                "  1. Register at https://developer.nrel.gov/signup/\n"
                "  2. Export data and place under data/raw/nrel_pvdaq/\n"
                "  3. See docs/datasets.md for expected structure."
            )
            return False
        return True

    def iter_samples(self) -> Iterator[RawSample]:
        root = self._raw_root
        if not root.exists():
            return
        import pandas as pd
        for csv_path in sorted(root.rglob("*.csv")):
            try:
                df = pd.read_csv(csv_path, nrows=500)
                # Check for irradiance / temperature columns
                irr_col = next((c for c in df.columns if "irradiance" in c.lower() or "ghi" in c.lower() or "poa" in c.lower()), None)
                temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
                wind_col = next((c for c in df.columns if "wind" in c.lower()), None)
                ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)

                for idx, row in df.iterrows():
                    env = {
                        "irradiance_w_m2": float(row[irr_col]) if irr_col and pd.notna(row[irr_col]) else None,
                        "ambient_temp_c": float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else None,
                        "wind_m_s": float(row[wind_col]) if wind_col and pd.notna(row[wind_col]) else None,
                    }
                    ts = str(row[ts_col]) if ts_col and pd.notna(row[ts_col]) else None
                    yield RawSample(
                        original_id=f"{csv_path.stem}_{idx}",
                        raw_fields={
                            "filepath": str(csv_path),
                            "native_label": "Healthy",
                            "environment": env,
                            "timestamp": ts,
                        },
                    )
            except Exception as e:
                logger.warning(f"Error reading {csv_path}: {e}")
                yield RawSample(
                    original_id=csv_path.stem,
                    raw_fields={"filepath": str(csv_path), "native_label": "Healthy"},
                )

    def to_canonical(self, raw: RawSample) -> CanonicalSample:
        return CanonicalSample(
            sample_id=self.make_sample_id(raw.original_id),
            domain_id=DOMAIN_ID,
            site_id=DOMAIN_ID,
            fault_label=["Healthy"],
            panel_id=None,
            inspection_pass_id=None,
            timestamp=raw.raw_fields.get("timestamp"),
            rgb_path=None,
            thermal_path=None,
            electrical_path=raw.raw_fields.get("electrical_path"),
            environment=raw.raw_fields.get("environment"),
            split_hint=None,
            metadata={
                "license": LICENSE,
                "original_filename": raw.raw_fields.get("filepath", ""),
                "native_label": "Healthy",
                "dataset": DATASET_ID,
            },
        )


def _load_label_map() -> dict:
    try:
        import yaml
        from src.common.paths import label_maps_dir
        path = label_maps_dir() / "nrel_pvdaq.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
