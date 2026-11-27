"""
src/data/adapters/own_site.py
Generic adapter for user-collected MVE-tier real-world data.

Dataset: Own-site collected data (any number of sites)
Domain ID: own_site_<n> (e.g. own_site_a, own_site_b)
Modality: RGB + Thermal (electrical optional via spot-check multimeter validation)
Access: MANUAL (physical collection by user)
License: User's own data.

Usage: Place collected data under data/raw/own_site_<n>/ following the
structure documented in docs/data.md. This adapter ingests it without
any special-cased code in the model/training layer.

Expected raw directory structure (docs/data.md will describe this in detail):
  data/raw/own_site_a/
    metadata.yaml          # site_id, collection_date, camera_model, gps, etc.
    rgb/<panel_id>/<timestamp>.jpg
    thermal/<panel_id>/<timestamp>.jpg
    electrical/<panel_id>/<timestamp>.npz  (optional)
    labels.csv             # panel_id, timestamp, fault_label(s)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from src.common.paths import data_raw
from src.data.adapters.base import CanonicalSample, DatasetAdapter, RawSample

logger = logging.getLogger(__name__)


class OwnSiteAdapter(DatasetAdapter):
    """Generic adapter for user-collected MVE-tier site data."""

    def __init__(self, site_id: str, raw_root: Path | None = None, label_map: dict | None = None):
        self._site_id = site_id
        self._raw_root = raw_root or data_raw(site_id)
        self._label_map = label_map or _load_label_map()

    def dataset_id(self) -> str:
        return self._site_id  # e.g. "own_site_a"

    def license_tag(self) -> str:
        return "User-collected data"

    def validate_raw(self) -> bool:
        root = self._raw_root
        if not root.exists():
            logger.warning(f"Own-site directory not found: {root}. Physical collection required.")
            return False
        labels_file = root / "labels.csv"
        if not labels_file.exists():
            logger.warning(f"Own-site labels.csv not found: {labels_file}")
            return False
        return True

    def iter_samples(self) -> Iterator[RawSample]:
        root = self._raw_root
        labels_file = root / "labels.csv"
        if not labels_file.exists():
            return
        with open(labels_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield RawSample(
                    original_id=f"{row.get('panel_id', 'unk')}_{row.get('timestamp', 'unk')}",
                    raw_fields=row,
                )

    def to_canonical(self, raw: RawSample) -> CanonicalSample:
        row = raw.raw_fields
        fault_labels_raw = row.get("fault_label", "Healthy").split("|")
        canonical_labels = []
        for lbl in fault_labels_raw:
            mapped = self._label_map.get(lbl.strip().lower(), lbl.strip())
            canonical_labels.append(mapped if isinstance(mapped, str) else mapped[0])
        if not canonical_labels:
            canonical_labels = ["Healthy"]

        return CanonicalSample(
            sample_id=self.make_sample_id(raw.original_id),
            domain_id=self._site_id,
            site_id=self._site_id,
            fault_label=canonical_labels,
            panel_id=row.get("panel_id"),
            inspection_pass_id=row.get("inspection_pass_id"),
            rgb_path=row.get("rgb_path"),
            thermal_path=row.get("thermal_path"),
            electrical_path=row.get("electrical_path"),
            environment={
                "irradiance_w_m2": _parse_float(row.get("irradiance_w_m2")),
                "ambient_temp_c": _parse_float(row.get("ambient_temp_c")),
                "wind_m_s": _parse_float(row.get("wind_m_s")),
            },
            split_hint=None,
            metadata={
                "license": "User-collected data",
                "original_filename": str(raw.raw_fields),
                "dataset": self._site_id,
            },
        )


def _parse_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _load_label_map() -> dict:
    try:
        import yaml
        from src.common.paths import label_maps_dir
        path = label_maps_dir() / "own_site.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
