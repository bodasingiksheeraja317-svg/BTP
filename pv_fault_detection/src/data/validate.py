"""
src/data/validate.py
Canonical schema validation for CanonicalSample records and Parquet index files.

Pure validation logic — no side effects, no file writes.
Raises ValidationError with specific field-level messages on schema violations.

Spec reference: Section 7 (canonical schema).
"""
from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

from src.data.adapters.base import CANONICAL_FAULT_CLASSES, CanonicalSample


class ValidationError(ValueError):
    """Raised when a CanonicalSample or Parquet row fails schema validation."""


VALID_SPLIT_HINTS = {"train", "val", "test", None}
VALID_ENVIRONMENT_KEYS = {"irradiance_w_m2", "ambient_temp_c", "wind_m_s"}


def validate_canonical_sample(sample: CanonicalSample) -> list[str]:
    """Validate one CanonicalSample against the Section 7 schema.

    Returns:
        List of error strings (empty = valid).
    """
    errors: list[str] = []

    # --- Required string fields ---
    if not sample.sample_id:
        errors.append("sample_id must be non-empty")
    elif "__" not in sample.sample_id:
        errors.append(f"sample_id '{sample.sample_id}' must contain '__' separator (format: domain_id__original_id)")

    if not sample.domain_id:
        errors.append("domain_id must be non-empty")

    if not sample.site_id:
        errors.append("site_id must be non-empty")

    # --- fault_label ---
    if not sample.fault_label:
        errors.append("fault_label must be a non-empty list")
    else:
        for lbl in sample.fault_label:
            if lbl not in CANONICAL_FAULT_CLASSES:
                errors.append(
                    f"fault_label '{lbl}' is not a canonical class. "
                    f"Allowed: {CANONICAL_FAULT_CLASSES}"
                )

    # --- split_hint ---
    if sample.split_hint not in VALID_SPLIT_HINTS:
        errors.append(f"split_hint must be 'train', 'val', 'test', or None; got '{sample.split_hint}'")

    # --- environment ---
    if sample.environment is not None:
        unexpected_keys = set(sample.environment.keys()) - VALID_ENVIRONMENT_KEYS
        if unexpected_keys:
            errors.append(f"environment has unexpected keys: {unexpected_keys}")
        # Values must be float or None
        for key in VALID_ENVIRONMENT_KEYS:
            val = sample.environment.get(key)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(f"environment.{key} must be float or None, got {type(val)}")

    # --- metadata ---
    if not isinstance(sample.metadata, dict):
        errors.append("metadata must be a dict")
    elif "license" not in sample.metadata:
        errors.append("metadata must contain 'license' key")

    return errors


def validate_parquet_row(row: dict) -> list[str]:
    """Validate one row from the canonical Parquet index.

    Args:
        row: dict representation of a Parquet row.

    Returns:
        List of error strings.
    """
    errors: list[str] = []
    required_fields = ["sample_id", "domain_id", "site_id", "fault_label"]
    for field in required_fields:
        if field not in row or not row[field]:
            errors.append(f"Missing required field: {field}")

    if "fault_label" in row:
        fault_labels = row["fault_label"]
        if isinstance(fault_labels, str):
            # Parquet may store as JSON string
            import json
            try:
                fault_labels = json.loads(fault_labels)
            except Exception:
                fault_labels = [fault_labels]
        for lbl in fault_labels:
            if lbl not in CANONICAL_FAULT_CLASSES:
                errors.append(f"fault_label '{lbl}' not in canonical taxonomy")

    return errors


def validate_full_index(index_path: str) -> tuple[int, int, list[str]]:
    """Validate every row in the canonical Parquet index.

    Args:
        index_path: Path to the full.parquet index file.

    Returns:
        Tuple of (total_rows, error_count, [error_messages]).
    """
    try:
        import pandas as pd
        df = pd.read_parquet(index_path)
    except Exception as e:
        return 0, 1, [f"Failed to read Parquet index: {e}"]

    all_errors: list[str] = []
    for i, row in df.iterrows():
        row_errors = validate_parquet_row(row.to_dict())
        for err in row_errors:
            all_errors.append(f"Row {i}: {err}")

    return len(df), len(all_errors), all_errors
