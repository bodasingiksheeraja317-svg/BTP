"""
tests/unit/test_electrical_repr.py
Unit tests for electrical representation: I-V resampling, P-V derivation, scalar extraction, STC normalization.
"""
from pathlib import Path
import numpy as np
import pytest

from src.preprocessing.electrical import (
    derive_pv_curve,
    extract_scalar_features,
    load_electrical_record,
    normalize_to_stc,
    preprocess_iv_curve,
)

pytestmark = pytest.mark.unit


def _create_mock_iv_curve():
    """Create a realistic synthetic I-V curve for a Si module (Voc~38V, Isc~9A)."""
    v = np.linspace(0, 38.0, 50)
    # Simple diode approximation
    i = 9.0 * (1.0 - np.exp((v - 38.0) / 1.5))
    i = np.clip(i, 0.0, 9.0)
    return np.stack([v, i], axis=1).astype(np.float32)


def test_iv_resampling_to_100_points():
    """Verify resampling transforms arbitrary curve length to exactly [100, 2]."""
    raw_iv = _create_mock_iv_curve()  # 50 points
    resampled = preprocess_iv_curve(raw_iv, target_points=100)
    assert resampled.shape == (100, 2)
    assert resampled.dtype == np.float32
    assert resampled[0, 0] == pytest.approx(0.0, abs=1e-3)  # V starts at 0
    assert resampled[-1, 0] == pytest.approx(38.0, abs=1e-3)  # V ends at Voc


def test_pv_curve_derivation():
    """Verify P-V curve derivation has shape [100, 1] with correct power values."""
    raw_iv = _create_mock_iv_curve()
    resampled_iv = preprocess_iv_curve(raw_iv, target_points=100)
    pv = derive_pv_curve(resampled_iv)
    assert pv.shape == (100, 1)
    # Power = V * I
    expected_power = resampled_iv[:, 0:1] * resampled_iv[:, 1:2]
    np.testing.assert_allclose(pv, expected_power, rtol=1e-4)


def test_scalar_features_extraction():
    """Verify extraction of 7 scalar features (Voc, Isc, Vmp, Imp, FF, Rs, Rsh)."""
    raw_iv = _create_mock_iv_curve()
    resampled_iv = preprocess_iv_curve(raw_iv, target_points=100)
    scalars = extract_scalar_features(resampled_iv)
    assert scalars.shape == (7,)
    voc, isc, vmp, imp, ff, rs, rsh = scalars

    assert 35.0 <= voc <= 40.0
    assert 8.0 <= isc <= 10.0
    assert 0.0 < vmp < voc
    assert 0.0 < imp < isc
    assert 0.5 <= ff <= 0.9  # Fill factor for valid solar cell
    assert 0.0 < rs < 10.0   # Physical Rs for module
    assert rsh > 50.0        # Physical Rsh for module


def test_rs_rsh_extraction_known_diode_curves():
    """Verify Rs and Rsh extraction against 3 known-analytic single-diode curves from pvlib."""
    import pvlib

    # 3 distinct test configurations: (Rs_true, Rsh_true)
    test_cases = [
        {"Rs_true": 0.35, "Rsh_true": 750.0, "IL": 9.2, "I0": 1e-9, "nNs": 2.0},
        {"Rs_true": 0.65, "Rsh_true": 350.0, "IL": 8.8, "I0": 2e-9, "nNs": 2.1},
        {"Rs_true": 1.20, "Rsh_true": 180.0, "IL": 9.0, "I0": 1e-9, "nNs": 2.0},
    ]

    for tc in test_cases:
        sd = pvlib.pvsystem.singlediode(
            photocurrent=tc["IL"],
            saturation_current=tc["I0"],
            resistance_series=tc["Rs_true"],
            resistance_shunt=tc["Rsh_true"],
            nNsVth=tc["nNs"],
        )
        voc = float(sd["v_oc"])
        v = np.linspace(0, voc, 100)
        i = pvlib.singlediode.bishop88_i_from_v(
            v,
            photocurrent=tc["IL"],
            saturation_current=tc["I0"],
            resistance_series=tc["Rs_true"],
            resistance_shunt=tc["Rsh_true"],
            nNsVth=tc["nNs"],
        )
        iv_curve = np.stack([v, i], axis=1).astype(np.float32)

        scalars = extract_scalar_features(iv_curve)
        voc_e, isc_e, vmp_e, imp_e, ff_e, rs_e, rsh_e = scalars

        # Rsh is estimated directly from linear slope at short-circuit (V~0): within 5% tolerance
        assert rsh_e == pytest.approx(tc["Rsh_true"], rel=0.05), (
            f"Extracted Rsh={rsh_e:.2f} deviated from True Rsh={tc['Rsh_true']:.2f}"
        )

        # Rs is estimated near Voc: captures series resistance + dynamic diode resistance; monotonic with Rs_true
        assert rs_e > tc["Rs_true"] * 0.8
        assert rs_e < tc["Rs_true"] * 4.0


def test_stc_normalization():
    """Verify I-V curve current scaling under non-STC irradiance."""
    raw_iv = _create_mock_iv_curve()
    # At 500 W/m2, current should be scaled by 1000/500 = 2.0 to normalize to STC
    normalized = normalize_to_stc(raw_iv, irradiance=500.0, temperature=25.0)
    assert normalized.shape == raw_iv.shape
    assert normalized[0, 1] == pytest.approx(raw_iv[0, 1] * 2.0, rel=1e-3)


def test_npz_load_roundtrip(tmp_path):
    """Verify .npz saving and load_electrical_record roundtrip."""
    iv = _create_mock_iv_curve()
    pv = derive_pv_curve(iv)
    scalars = extract_scalar_features(iv)

    npz_path = tmp_path / "test_panel.npz"
    np.savez(str(npz_path), iv_curve=iv, pv_curve=pv, scalars=scalars)

    record = load_electrical_record(npz_path)
    assert record is not None
    assert "iv_curve" in record
    assert "pv_curve" in record
    assert "scalars" in record
    np.testing.assert_allclose(record["iv_curve"], iv)
