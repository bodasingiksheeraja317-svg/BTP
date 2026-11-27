"""
src/preprocessing/electrical.py
Electrical I-V/P-V curve preprocessing pipeline.

Pure-function library. No YAML reading.
Spec reference: Section 9.3, Section 5 (electrical branch spec)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# STC (Standard Test Conditions) defaults — config-overridable
STC_IRRADIANCE = 1000.0   # W/m^2
STC_TEMPERATURE = 25.0    # Celsius
IV_RESAMPLE_POINTS = 100  # per spec Section 5
SCALAR_FEATURE_NAMES = ["Voc", "Isc", "Vmp", "Imp", "FF", "Rs", "Rsh"]


def load_electrical_record(electrical_path: str | Path) -> dict | None:
    """Load a serialized electrical record (.npz).

    Expected keys in .npz: iv_curve, pv_curve, scalars, (optional: generation_params)
    Returns dict with arrays or None if load fails.
    """
    path = Path(electrical_path)
    if not path.exists():
        logger.warning(f"Electrical record not found: {path}")
        return None
    try:
        data = np.load(str(path), allow_pickle=True)
        return dict(data)
    except Exception as e:
        logger.error(f"Failed to load electrical record {path}: {e}")
        return None


def preprocess_iv_curve(
    iv_curve: np.ndarray,
    target_points: int = IV_RESAMPLE_POINTS,
) -> np.ndarray:
    """Resample I-V curve to a fixed number of points.

    Args:
        iv_curve: Raw I-V curve, shape [N, 2] (columns: voltage, current) or [2, N].
        target_points: Number of output points (100 per spec).

    Returns:
        Resampled curve as float32 [target_points, 2] (V, I).
    """
    from scipy.interpolate import interp1d

    if iv_curve.ndim == 2 and iv_curve.shape[1] == 2:
        v_raw, i_raw = iv_curve[:, 0], iv_curve[:, 1]
    elif iv_curve.ndim == 2 and iv_curve.shape[0] == 2:
        v_raw, i_raw = iv_curve[0], iv_curve[1]
    else:
        raise ValueError(f"iv_curve must be [N,2] or [2,N], got {iv_curve.shape}")

    # Remove duplicate voltage values
    _, unique_idx = np.unique(v_raw, return_index=True)
    v_raw, i_raw = v_raw[unique_idx], i_raw[unique_idx]

    if len(v_raw) < 2:
        logger.warning("IV curve has fewer than 2 unique points; returning zeros")
        return np.zeros((target_points, 2), dtype=np.float32)

    v_new = np.linspace(v_raw.min(), v_raw.max(), target_points)
    interp_i = interp1d(v_raw, i_raw, kind="linear", fill_value="extrapolate")
    i_new = interp_i(v_new)

    return np.stack([v_new, i_new], axis=1).astype(np.float32)  # [100, 2]


def derive_pv_curve(iv_curve: np.ndarray) -> np.ndarray:
    """Derive P-V curve from I-V curve.

    Args:
        iv_curve: [N, 2] (V, I) array.

    Returns:
        P-V curve [N, 1] (power = V * I).
    """
    v = iv_curve[:, 0]
    i = iv_curve[:, 1]
    p = v * i
    return p.reshape(-1, 1).astype(np.float32)  # [N, 1]


def extract_scalar_features(iv_curve: np.ndarray) -> np.ndarray:
    """Extract scalar PV module parameters from I-V curve.

    Features: Voc, Isc, Vmp, Imp, FF, Rs (approx), Rsh (approx)
    These are stored in the .npz electrical record alongside the curve.

    Returns:
        float32 array of shape [7] with values in order: Voc, Isc, Vmp, Imp, FF, Rs, Rsh
    """
    v = iv_curve[:, 0]
    i = iv_curve[:, 1]
    p = v * i

    voc = float(v.max())
    isc = float(i[np.argmin(np.abs(v))])  # I at V~0
    vmp_idx = np.argmax(p)
    vmp = float(v[vmp_idx])
    imp = float(i[vmp_idx])
    pmp = float(p[vmp_idx])

    ff = pmp / (voc * isc) if (voc * isc) > 0 else 0.0

    rs = _estimate_rs(v, i)
    rsh = _estimate_rsh(v, i)

    return np.array([voc, isc, vmp, imp, ff, rs, rsh], dtype=np.float32)


def _estimate_rs(v: np.ndarray, i: np.ndarray, n_points: int = 8) -> float:
    """Estimate series resistance from slope near open-circuit (Voc, high voltage / low current)."""
    try:
        from scipy.stats import linregress
        order = np.argsort(v)
        v_sorted, i_sorted = v[order], i[order]
        # Points near Voc where current is dropping towards zero
        pos_idx = np.where(i_sorted >= 0.0)[0]
        if len(pos_idx) >= 3:
            pts = min(n_points, len(pos_idx))
            v_sub = v_sorted[pos_idx[-pts:]]
            i_sub = i_sorted[pos_idx[-pts:]]
            slope, _, _, _, _ = linregress(v_sub, i_sub)
            if slope < 0:
                return float(-1.0 / slope)
        return 0.0
    except Exception:
        return 0.0


def _estimate_rsh(v: np.ndarray, i: np.ndarray, n_points: int = 8) -> float:
    """Estimate shunt resistance from slope near short-circuit (V~0, low voltage / high current)."""
    try:
        from scipy.stats import linregress
        order = np.argsort(v)
        v_sorted, i_sorted = v[order], i[order]
        pts = min(n_points, len(v_sorted))
        if pts >= 3:
            v_sub = v_sorted[:pts]
            i_sub = i_sorted[:pts]
            slope, _, _, _, _ = linregress(v_sub, i_sub)
            if slope < -1e-6:
                return float(-1.0 / slope)
        return 1e5
    except Exception:
        return 1e5


def normalize_to_stc(
    iv_curve: np.ndarray,
    irradiance: float | None,
    temperature: float | None,
    stc_irradiance: float = STC_IRRADIANCE,
    stc_temperature: float = STC_TEMPERATURE,
) -> np.ndarray:
    """Normalize I-V curve to Standard Test Conditions."""
    if irradiance is None or temperature is None:
        return iv_curve

    irr_ratio = stc_irradiance / max(irradiance, 1.0)
    iv_corrected = iv_curve.copy()
    iv_corrected[:, 1] *= irr_ratio
    return iv_corrected
