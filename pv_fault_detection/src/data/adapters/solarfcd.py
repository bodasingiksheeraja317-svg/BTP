"""
src/data/adapters/solarfcd.py
Adapter for the SolarFCD dataset (RGB + Thermal unified, multi-source).

Dataset: SolarFCD
Domain ID: solarfcd
Modality: RGB + Thermal (unified)
Access: MANUAL (arXiv/dataset-repo release — TO VERIFY current hosting)
License: Open — check dataset repo. TO VERIFY.
Note: PVMD is folded into SolarFCD per spec Section 6.2.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from src.common.paths import data_raw
from src.data.adapters.base import CanonicalSample, DatasetAdapter, RawSample

logger = logging.getLogger(__name__)

DATASET_ID = "solarfcd"
DOMAIN_ID = "solarfcd"
LICENSE = "Open — TO VERIFY exact terms at dataset repository"


class SolarFCDAdapter(DatasetAdapter):
    """SolarFCD dataset adapter (MANUAL download required)."""

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
            logger.warning(f"SolarFCD raw directory not found: {root}. MANUAL download required.")
            return False
        return True

    def iter_samples(self) -> Iterator[RawSample]:
        root = self._raw_root
        if not root.exists():
            return
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_paths = sorted({p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts})
        for img_path in image_paths:
            yield RawSample(
                original_id=img_path.stem,
                raw_fields={
                    "filepath": str(img_path),
                    "native_label": img_path.parent.name,
                    "filename": img_path.name,
                },
            )

    def to_canonical(self, raw: RawSample) -> CanonicalSample:
        native_label = raw.raw_fields.get("native_label", "")
        canonical_labels = self._label_map.get(native_label.lower(), ["Healthy"])
        if isinstance(canonical_labels, str):
            canonical_labels = [canonical_labels]
        return CanonicalSample(
            sample_id=self.make_sample_id(raw.original_id),
            domain_id=DOMAIN_ID,
            site_id=DOMAIN_ID,
            fault_label=canonical_labels,
            panel_id=None,
            inspection_pass_id=None,
            timestamp=None,
            rgb_path=raw.raw_fields.get("rgb_path"),
            thermal_path=raw.raw_fields.get("thermal_path", raw.raw_fields.get("filepath")),
            electrical_path=None,
            environment=None,
            split_hint=None,
            metadata={
                "license": LICENSE,
                "original_filename": raw.raw_fields.get("filename", ""),
                "native_label": native_label,
                "dataset": DATASET_ID,
            },
        )


def _load_label_map() -> dict:
    try:
        import yaml
        from src.common.paths import label_maps_dir
        path = label_maps_dir() / "solarfcd.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
