"""
src/preprocessing/synchronization.py
UTC timestamp alignment for image <-> electrical data pairing.

Spec Section 9.3:
  - Match image and electrical timestamps within +/-2 minutes (120 seconds) UTC.
  - Unmatched images get electrical_path=null in the canonical index.

Pure-function library.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_DELTA_SECONDS = 120  # +/-2 min per spec


def parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp string to a timezone-aware datetime."""
    if ts is None:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as e:
        logger.warning(f"Could not parse timestamp '{ts}': {e}")
        return None


def match_timestamps(
    image_records: list[dict],
    electrical_records: list[dict],
    max_delta_seconds: int = DEFAULT_MAX_DELTA_SECONDS,
    image_ts_key: str = "timestamp",
    electrical_ts_key: str = "timestamp",
    electrical_id_key: str = "electrical_path",
) -> list[dict]:
    """Match image records to electrical records by closest timestamp within tolerance."""
    elec_parsed = []
    for rec in electrical_records:
        dt = parse_iso_timestamp(rec.get(electrical_ts_key))
        if dt is not None:
            elec_parsed.append((dt, rec.get(electrical_id_key)))

    matched = 0
    unmatched = 0
    result = []
    for img_rec in image_records:
        img_dt = parse_iso_timestamp(img_rec.get(image_ts_key))
        rec_copy = dict(img_rec)
        rec_copy["electrical_path"] = None

        if img_dt is None or not elec_parsed:
            unmatched += 1
            result.append(rec_copy)
            continue

        best_delta = float("inf")
        best_path = None
        for elec_dt, elec_path in elec_parsed:
            delta = abs((img_dt - elec_dt).total_seconds())
            if delta < best_delta:
                best_delta = delta
                best_path = elec_path

        if best_delta <= max_delta_seconds:
            rec_copy["electrical_path"] = best_path
            matched += 1
        else:
            unmatched += 1

        result.append(rec_copy)

    logger.info(
        f"Timestamp matching: {matched} matched, {unmatched} unmatched "
        f"(tolerance: +/-{max_delta_seconds}s)"
    )
    return result
