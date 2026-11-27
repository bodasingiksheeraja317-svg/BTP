"""
src/data/adapters/elpv.py
Adapter for the ELPV / E-ELPV electroluminescence dataset.

Dataset: ELPV
Domain ID: elpv
Modality: EL (electroluminescence images)
Access: AUTOMATIC (public GitHub: zae-bayern/elpv-dataset)
License: Open, citation required.

IMPORTANT: ELPV is excluded from the DEFAULT adapter registry (spec Section 6.2).
It is an optional sub-study only. Never part of the core field-system fault taxonomy.
Only included when indexer is run with --include-el-substudy.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from src.common.paths import data_raw
from src.data.adapters.base import CanonicalSample, DatasetAdapter, RawSample

logger = logging.getLogger(__name__)

DATASET_ID = "elpv"
DOMAIN_ID = "elpv"
LICENSE = "Open dataset — cite: Deitsch et al. (2019), zae-bayern/elpv-dataset"


class ELPVAdapter(DatasetAdapter):
    """ELPV electroluminescence dataset adapter (optional sub-study)."""

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
            logger.warning(f"ELPV raw directory not found: {root}")
            return False
        labels_files = list(root.rglob("labels.csv"))
        if not labels_files:
            logger.warning(f"ELPV labels.csv not found under {root}")
            return False
        logger.info(f"ELPV: validated at {root}")
        return True

    def iter_samples(self) -> Iterator[RawSample]:
        root = self._raw_root
        labels_files = list(root.rglob("labels.csv"))
        if not labels_files:
            logger.error(f"Cannot iterate ELPV samples: labels.csv not found")
            return
        labels_file = labels_files[0]
        base_dir = labels_file.parent
        import csv
        with open(labels_file) as f:
            # Check if headers exist, elpv format is often space or semicolon separated
            first_line = f.readline()
            f.seek(0)
            delimiter = ";" if ";" in first_line else " " if " " in first_line else ","
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    rel_img = parts[0].strip()
                    prob = parts[1].strip()
                    val = float(prob) if prob.replace(".", "", 1).isdigit() else 0.0
                    native_lbl = "defective" if val > 0.5 else "functional"
                    img_path = base_dir / rel_img
                    if img_path.exists():
                        yield RawSample(
                            original_id=img_path.stem,
                            raw_fields={
                                "filepath": str(img_path),
                                "native_label": native_lbl,
                                "probability": str(val),
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
            rgb_path=None,      # EL images mapped as single-channel thermal
            thermal_path=raw.raw_fields.get("filepath"),
            electrical_path=None,
            environment=None,
            split_hint=None,
            metadata={
                "license": LICENSE,
                "original_filename": raw.raw_fields.get("filepath", ""),
                "native_label": native_label,
                "probability": raw.raw_fields.get("probability", ""),
                "dataset": DATASET_ID,
                "note": "EL sub-study only — excluded from core taxonomy experiments",
            },
        )


def _load_label_map() -> dict:
    try:
        import yaml
        from src.common.paths import label_maps_dir
        path = label_maps_dir() / "elpv.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load ELPV label map: {e}")
        return {}
