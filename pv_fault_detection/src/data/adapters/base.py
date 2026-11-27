"""
src/data/adapters/base.py
Abstract DatasetAdapter interface — the ONLY place dataset-specific logic lives.

Every public dataset and every own-collected dataset implements this interface.
The rest of the system (preprocessing, splitting, models) never has dataset-specific
branches outside src/data/adapters/.

Spec reference: Section 9.1
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Fault taxonomy (FIXED — spec Section 7.1 — do NOT modify)
# ---------------------------------------------------------------------------
CANONICAL_FAULT_CLASSES = [
    "Healthy",
    "Hotspot",
    "Open/Short-circuit",
    "Shading/Obstruction",
    "Soiling",
    "Structural (visual)",
    "Electrical/Diode fault",
]


# ---------------------------------------------------------------------------
# Data schemas
# ---------------------------------------------------------------------------

@dataclass
class RawSample:
    """Source-native record, before canonical mapping. Fields are dataset-specific."""
    original_id: str
    raw_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalSample:
    """Fully-resolved record conforming to spec Section 7 schema.

    Every field must match the schema exactly. Adapters must fill all REQUIRED
    fields; optional fields may be None where the source dataset doesn't provide them.
    """
    # --- REQUIRED ---
    sample_id: str              # globally unique: "{domain_id}__{original_id}"
    domain_id: str              # one of the domain-ID prefixes in spec Section 6.2
    site_id: str                # physical site; may equal domain_id for single-site datasets
    fault_label: list[str]      # multi-label; min length 1; subset of CANONICAL_FAULT_CLASSES

    # --- CONDITIONAL REQUIRED ---
    panel_id: str | None = None             # required for real-collected; null OK for public
    inspection_pass_id: str | None = None   # groups same-flight images
    timestamp: str | None = None            # ISO-8601 UTC; null if source has none

    # --- MODALITY PATHS (null = structurally absent) ---
    rgb_path: str | None = None             # relative under data/processed/
    thermal_path: str | None = None
    thermal_raw_path: str | None = None     # .npy radiometric temp array
    electrical_path: str | None = None      # .npz: iv_curve, pv_curve, scalars

    # --- ENVIRONMENT (individual fields may be null) ---
    environment: dict | None = None         # {irradiance_w_m2, ambient_temp_c, wind_m_s}

    # --- SPLIT HINT ---
    split_hint: str | None = None           # "train"|"val"|"test" from official split only

    # --- METADATA (free-form passthrough) ---
    metadata: dict = field(default_factory=dict)
    # metadata must include: {license: str, original_filename: str, dataset: str}

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """Basic schema validation. Full validation is in src/data/validate.py."""
        assert self.sample_id, "sample_id must be non-empty"
        assert self.domain_id, "domain_id must be non-empty"
        assert self.site_id, "site_id must be non-empty"
        assert self.fault_label, "fault_label must be non-empty list"
        for label in self.fault_label:
            assert label in CANONICAL_FAULT_CLASSES, (
                f"Unknown fault label '{label}'. Must be one of: {CANONICAL_FAULT_CLASSES}"
            )
        if self.split_hint is not None:
            assert self.split_hint in {"train", "val", "test"}, (
                f"split_hint must be 'train', 'val', or 'test', got '{self.split_hint}'"
            )


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DatasetAdapter(ABC):
    """Abstract base class for all dataset adapters.

    Every dataset (public or own-collected) implements this interface.
    The indexer (src/data/indexer.py) is the ONLY caller.
    """

    @abstractmethod
    def dataset_id(self) -> str:
        """Return the canonical dataset ID prefix (e.g. 'irsm', 'cali')."""

    @abstractmethod
    def validate_raw(self) -> bool:
        """Check data/raw/<dataset_id>/ has expected structure.

        Returns:
            True if the raw directory exists and has the expected structure.
            False (NOT raises) on mismatch — callers report status, they don't crash.
        """

    @abstractmethod
    def iter_samples(self) -> Iterator[RawSample]:
        """Yield one RawSample per source-native sample, before canonical mapping."""

    @abstractmethod
    def to_canonical(self, raw_sample: RawSample) -> CanonicalSample:
        """Apply label_maps/<dataset_id>.yaml and fill the Section 7 schema."""

    @abstractmethod
    def license_tag(self) -> str:
        """Return the dataset license string (recorded in metadata.license)."""

    # ------------------------------------------------------------------
    # Non-abstract helpers all adapters get for free
    # ------------------------------------------------------------------

    def make_sample_id(self, original_id: str) -> str:
        """Construct globally-unique sample_id from domain_id + original_id."""
        return f"{self.dataset_id()}__{original_id}"

    def is_available(self) -> bool:
        """True if the raw dataset directory exists and validate_raw() passes."""
        return self.validate_raw()

    def status(self) -> str:
        """Return a human-readable status string for DATASET_STATUS.md."""
        if self.is_available():
            return "PRESENT"
        return "MISSING"
