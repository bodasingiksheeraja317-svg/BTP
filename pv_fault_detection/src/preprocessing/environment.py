"""
src/preprocessing/environment.py
Environmental feature cleaning and null-handling.

Key design decision (spec Section 7):
  NULL environment values use a learned 'unknown' embedding token FLAG,
  NOT zero-imputation. Zero has physical meaning here (zero irradiance = valid).

This module returns cleaned numerical values + a flag array indicating which
fields are null (to be converted to the learned unknown embedding by the model).

Pure-function library.
"""
from __future__ import annotations

import numpy as np

IDX_IRRADIANCE = 0
IDX_TEMP = 1
IDX_WIND = 2
ENV_DIM = 3

IRRADIANCE_MIN = 0.0
IRRADIANCE_MAX = 1500.0   # W/m^2
TEMP_MIN = -40.0           # Celsius
TEMP_MAX = 80.0
WIND_MIN = 0.0             # m/s
WIND_MAX = 50.0


def process_environment(env_dict: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Clean and vectorize environment dict.

    Args:
        env_dict: {irradiance_w_m2: float|None, ambient_temp_c: float|None, wind_m_s: float|None}
                  or None (entire environment absent).

    Returns:
        Tuple of:
          - values: float32 array [3] — cleaned numerical values (0.0 where null/missing)
          - known_mask: bool array [3] — True where value is known (not null)
                        The model substitutes learned 'unknown' embeddings where False.
    """
    if env_dict is None:
        return np.zeros(ENV_DIM, dtype=np.float32), np.zeros(ENV_DIM, dtype=bool)

    irradiance = _clean(env_dict.get("irradiance_w_m2"), IRRADIANCE_MIN, IRRADIANCE_MAX)
    temp = _clean(env_dict.get("ambient_temp_c"), TEMP_MIN, TEMP_MAX)
    wind = _clean(env_dict.get("wind_m_s"), WIND_MIN, WIND_MAX)

    values = np.array(
        [irradiance[0], temp[0], wind[0]], dtype=np.float32
    )
    known_mask = np.array(
        [irradiance[1], temp[1], wind[1]], dtype=bool
    )
    return values, known_mask


def _clean(value, min_val: float, max_val: float) -> tuple[float, bool]:
    """Return (cleaned_value, is_known). Known=False if null or out of physical range."""
    if value is None:
        return 0.0, False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not (min_val <= v <= max_val):
        return 0.0, False
    return v, True
