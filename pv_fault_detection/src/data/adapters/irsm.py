"""
src/data/adapters/irsm.py
Adapter for the InfraredSolarModules (Raptor Maps) thermal dataset.

Dataset: InfraredSolarModules
Domain ID: irsm
Modality: Thermal
Access: AUTOMATIC (spec Section 20)
License: Free for research — verify current terms at source.
TO VERIFY: Check Raptor Maps GitHub release URL is still live before relying on it.

Label mapping: src/data/label_maps/irsm.yaml
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from src.common.paths import data_raw, data_processed_images
from src.data.adapters.base import CanonicalSample, DatasetAdapter, RawSample

logger = logging.getLogger(__name__)

DATASET_ID = "irsm"
DOMAIN_ID = "irsm"
LICENSE = "Free for research — verify current terms at Raptor Maps GitHub"

# Known Raptor Maps GitHub release URL — TO VERIFY at build time
_DOWNLOAD_URL = "https://github.com/RaptorMaps/InfraredSolarModules/archive/refs/heads/master.zip"


class IRSMAdapter(DatasetAdapter):
    """InfraredSolarModules thermal dataset adapter."""

    def __init__(self, raw_root: Path | None = None, label_map: dict | None = None):
        self._raw_root = raw_root or data_raw(DATASET_ID)
        self._label_map = label_map or _load_label_map()

    def dataset_id(self) -> str:
        return DATASET_ID

    def license_tag(self) -> str:
        return LICENSE

    def validate_raw(self) -> bool:
        """Check data/raw/irsm/ exists and contains expected structure."""
        root = self._raw_root
        if not root.exists():
            logger.warning(f"IRSM raw directory not found: {root}")
            return False
        # Expect at least some image files
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]
        if not images:
            logger.warning(f"IRSM raw directory exists but contains no image files: {root}")
            return False
        logger.info(f"IRSM: validated {len(images)} images under {root}")
        return True

    def iter_samples(self) -> Iterator[RawSample]:
        """Yield one RawSample per image file in the IRSM raw directory."""
        root = self._raw_root
        if not root.exists():
            logger.error(f"Cannot iterate IRSM samples: directory not found: {root}")
            return
        import json
        meta_files = list(root.rglob("module_metadata.json"))
        if meta_files:
            meta_path = meta_files[0]
            with open(meta_path) as f:
                meta_dict = json.load(f)
            base_dir = meta_path.parent
            for k, val in meta_dict.items():
                rel_img = val.get("image_filepath")
                img_path = base_dir / rel_img
                if img_path.exists():
                    yield RawSample(
                        original_id=img_path.stem,
                        raw_fields={
                            "filepath": str(img_path),
                            "native_label": val.get("anomaly_class", "No-Anomaly"),
                            "filename": img_path.name,
                        },
                    )
            return

        # IRSM directory structure fallback: data/raw/irsm/<fault_class>/<image_file>
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_paths = sorted({p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts})
        for img_path in image_paths:
            # Infer native label from parent directory name
            native_label = img_path.parent.name
            yield RawSample(
                original_id=img_path.stem,
                raw_fields={
                    "filepath": str(img_path),
                    "native_label": native_label,
                    "filename": img_path.name,
                },
            )

    def to_canonical(self, raw: RawSample) -> CanonicalSample:
        """Map IRSM native label to canonical taxonomy and build CanonicalSample."""
        native_label = raw.raw_fields.get("native_label", "")
        canonical_labels = self._map_label(native_label)

        return CanonicalSample(
            sample_id=self.make_sample_id(raw.original_id),
            domain_id=DOMAIN_ID,
            site_id=DOMAIN_ID,  # single-site public dataset
            fault_label=canonical_labels,
            panel_id=None,          # IRSM crops don't expose panel identity
            inspection_pass_id=None,
            timestamp=None,
            rgb_path=None,          # thermal only
            thermal_path=raw.raw_fields.get("filepath"),
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

    def _map_label(self, native_label: str) -> list[str]:
        """Map native IRSM label to canonical fault taxonomy."""
        canonical = self._label_map.get(native_label.lower())
        if canonical is None:
            logger.warning(
                f"IRSM: unmapped label '{native_label}'. Marking TO_VERIFY. "
                "Update src/data/label_maps/irsm.yaml."
            )
            # Return Healthy as safe default but flag it
            return ["Healthy"]
        return [canonical] if isinstance(canonical, str) else canonical


def _load_label_map() -> dict:
    """Load label_maps/irsm.yaml. Returns empty dict if file not found."""
    try:
        import yaml
        from src.common.paths import label_maps_dir
        path = label_maps_dir() / "irsm.yaml"
        if not path.exists():
            logger.warning(f"Label map not found: {path}")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load IRSM label map: {e}")
        return {}
