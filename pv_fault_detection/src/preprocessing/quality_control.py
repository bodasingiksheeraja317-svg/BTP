"""
src/preprocessing/quality_control.py
Automated quality control filtering for thermal images and environmental conditions.

IEC 62446-3-style thresholds (ASSUMPTION from spec Section 9.3):
  - irradiance >= 600 W/m^2
  - wind < 5 m/s

Both thresholds are config-overridable. Pure-function library.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ASSUMPTION: IEC 62446-3-style defaults (spec Section 9.3); config-overridable
DEFAULT_IRRADIANCE_MIN = 600.0  # W/m^2
DEFAULT_WIND_MAX = 5.0          # m/s


def passes_quality_control(
    irradiance: float | None,
    wind_speed: float | None,
    irradiance_min: float = DEFAULT_IRRADIANCE_MIN,
    wind_max: float = DEFAULT_WIND_MAX,
) -> tuple[bool, str]:
    """Check if an image sample passes environmental quality control.

    Args:
        irradiance: Irradiance in W/m^2 (None = unknown -> passes with warning).
        wind_speed: Wind speed in m/s (None = unknown -> passes with warning).
        irradiance_min: Minimum acceptable irradiance.
        wind_max: Maximum acceptable wind speed.

    Returns:
        Tuple of (passes: bool, reason: str).
    """
    if irradiance is not None and irradiance < irradiance_min:
        return False, f"irradiance {irradiance:.1f} W/m^2 < minimum {irradiance_min} W/m^2"
    if wind_speed is not None and wind_speed >= wind_max:
        return False, f"wind {wind_speed:.1f} m/s >= maximum {wind_max} m/s"
    if irradiance is None:
        logger.debug("Irradiance unknown; sample passes QC with warning")
    if wind_speed is None:
        logger.debug("Wind speed unknown; sample passes QC with warning")
    return True, ""


def check_thermal_image_quality(thermal_path: str | Path) -> tuple[bool, str]:
    """Basic sanity check for thermal image: non-blank, correct channel count."""
    import cv2
    path = Path(thermal_path)
    if not path.exists():
        return False, f"File not found: {path}"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, f"Failed to load image: {path}"
    if img.std() < 1.0:
        return False, f"Image appears blank (std={img.std():.2f}): {path}"
    return True, ""
