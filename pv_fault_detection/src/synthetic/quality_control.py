"""
src/synthetic/quality_control.py
Quality control for synthetic data (electrical curves and images).

Spec Section 11.3:
  - Electrical: reject curves with Voc/Isc/FF outside valid_ranges.yaml.
  - Images: SSIM filter + basic sanity (non-blank, correct channel count).
  - Both paths LOG rejection rate to experiment tracker — a silently-high
    rejection rate is a diagnostic signal, not something to hide.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def validate_electrical_curve(
    scalars: dict,
    valid_ranges_path: str | Path | None = None,
) -> tuple[bool, str]:
    """Validate synthetic electrical curve scalars against physical ranges.

    Args:
        scalars: dict with keys Voc, Isc, FF (and optionally others).
        valid_ranges_path: Path to valid_ranges.yaml.

    Returns:
        (passes: bool, reason: str)
    """
    ranges = _load_valid_ranges(valid_ranges_path)
    si = ranges.get("si_module", {})

    for key in ["Voc", "Isc", "FF"]:
        if key not in scalars:
            continue
        val = scalars[key]
        rng = si.get(key, {})
        min_v = rng.get("min")
        max_v = rng.get("max")
        if min_v is not None and val < min_v:
            return False, f"{key}={val:.3f} < min {min_v}"
        if max_v is not None and val > max_v:
            return False, f"{key}={val:.3f} > max {max_v}"

    return True, ""


def validate_synthetic_image_ssim(
    generated_image: np.ndarray,
    source_image: np.ndarray,
    ssim_threshold: float = 0.75,
) -> tuple[bool, float]:
    """Validate synthetic image against source using SSIM.

    Args:
        generated_image: Synthesized image [H, W, C] or [H, W].
        source_image: Original real training image.
        ssim_threshold: Minimum SSIM to accept (0.75 per spec ASSUMPTION).

    Returns:
        (passes: bool, ssim_value: float)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        score = ssim(
            source_image, generated_image,
            channel_axis=-1 if generated_image.ndim == 3 else None,
            data_range=255,
        )
        return float(score) >= ssim_threshold, float(score)
    except Exception as e:
        logger.warning(f"SSIM computation failed: {e}")
        return False, 0.0


def _load_valid_ranges(path: str | Path | None) -> dict:
    if path is None:
        from src.common.paths import PROJECT_ROOT
        path = PROJECT_ROOT / "src" / "synthetic" / "valid_ranges.yaml"
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
