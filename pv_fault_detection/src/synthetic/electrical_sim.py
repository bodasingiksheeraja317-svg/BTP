"""
src/synthetic/electrical_sim.py
pvlib-based electrical I-V/P-V curve fault simulator.

Spec Section 11.1:
  - Uses pvlib-python single/two-diode equivalent-circuit model.
  - NEVER hand-rolled diode equations (explicit spec decision).
  - Parameters perturbed per fault class from fault_injection_table.yaml.
  - Physical range validation via quality_control.py before acceptance.
  - Storage: data/synthetic/electrical/<fault_class>/<uuid>.npz
  - Ground-truth fault label set by construction — never inferred.

CLI: python -m src.synthetic.electrical_sim --config configs/base.yaml

Full implementation in Stage 11. Stub here for Stage 1 scaffolding.
"""
import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any
import numpy as np
import pvlib
import yaml

from src.common.paths import PROJECT_ROOT, data_synthetic_electrical
from src.preprocessing.electrical import (
    derive_pv_curve,
    extract_scalar_features,
    preprocess_iv_curve,
)
from src.synthetic.quality_control import validate_electrical_curve

logger = logging.getLogger(__name__)


def _load_fault_table() -> dict:
    table_path = PROJECT_ROOT / "src" / "synthetic" / "fault_injection_table.yaml"
    if not table_path.exists():
        return {}
    with open(table_path) as f:
        return yaml.safe_load(f) or {}


def generate_iv_curve_for_fault(
    fault_class: str,
    irradiance: float = 1000.0,
    temperature: float = 25.0,
    injection_params: dict | None = None,
    noise_std: float = 0.02,
    output_dir: Path | str | None = None,
    save: bool = False,
) -> dict | None:
    """Generate one synthetic I-V/P-V curve for a fault class using pvlib.

    Returns dict with: iv_curve, pv_curve, scalars, generation_params, fault_class, file_path
    or None if rejected by physical-range validation.
    """
    table = _load_fault_table()
    base_stc = table.get("base_params_stc", {
        "I_L_ref": 8.5,
        "I_o_ref": 1.0e-10,
        "R_s": 0.4,
        "R_sh_ref": 200.0,
        "a_ref": 1.6,
    })

    # Base STC parameters with optional noise
    I_L = float(base_stc.get("I_L_ref", 8.5)) * (irradiance / 1000.0) * (1.0 + np.random.normal(0, noise_std))
    I_o = float(base_stc.get("I_o_ref", 1.0e-10)) * (1.0 + np.random.normal(0, noise_std))
    R_s = float(base_stc.get("R_s", 0.4)) * (1.0 + np.random.normal(0, noise_std))
    R_sh = float(base_stc.get("R_sh_ref", 200.0)) * (1000.0 / max(irradiance, 100.0)) * (1.0 + np.random.normal(0, noise_std))
    a = float(base_stc.get("a_ref", 1.6)) * (1.0 + np.random.normal(0, noise_std))

    # Temperature effects
    T_K = temperature + 273.15
    a = a * (T_K / 298.15)
    I_L = I_L * (1.0 + 0.0005 * (temperature - 25.0))

    # Apply fault injection perturbations
    fault_info = table.get("fault_classes", {}).get(fault_class, {})
    perts = fault_info.get("perturbations", {})
    if injection_params:
        perts = injection_params

    if "R_s" in perts:
        if "multiplier" in perts["R_s"]:
            R_s *= perts["R_s"]["multiplier"]
        if "add" in perts["R_s"]:
            R_s += perts["R_s"]["add"]

    if "R_sh" in perts:
        if "multiplier" in perts["R_sh"]:
            R_sh *= perts["R_sh"]["multiplier"]
        if "add" in perts["R_sh"]:
            R_sh += perts["R_sh"]["add"]

    if "I_L" in perts:
        if "multiplier" in perts["I_L"]:
            I_L *= perts["I_L"]["multiplier"]
        if "add" in perts["I_L"]:
            I_L += perts["I_L"]["add"]

    # Clamp parameters to physically valid minimums
    I_L = max(I_L, 0.1)
    I_o = max(I_o, 1e-12)
    R_s = max(R_s, 0.01)
    R_sh = max(R_sh, 1.0)
    a = max(a, 0.5)

    try:
        sd = pvlib.pvsystem.singlediode(
            photocurrent=I_L,
            saturation_current=I_o,
            resistance_series=R_s,
            resistance_shunt=R_sh,
            nNsVth=a,
        )
        voc = float(sd["v_oc"])
        if voc <= 0 or np.isnan(voc):
            return None

        v = np.linspace(0, voc, 100)
        i = pvlib.singlediode.bishop88_i_from_v(
            v,
            photocurrent=I_L,
            saturation_current=I_o,
            resistance_series=R_s,
            resistance_shunt=R_sh,
            nNsVth=a,
        )
        if np.isnan(i).any():
            return None

        raw_iv = np.stack([v, i], axis=1).astype(np.float32)
        iv_curve = preprocess_iv_curve(raw_iv, target_points=100)
        pv_curve = derive_pv_curve(iv_curve)
        scalars = extract_scalar_features(iv_curve)

        voc_e, isc_e, vmp_e, imp_e, ff_e, rs_e, rsh_e = scalars

        # Validate physical validity
        scalar_dict = {
            "Voc": voc_e,
            "Isc": isc_e,
            "FF": ff_e,
            "Pmp": float(vmp_e * imp_e),
            "Rs": rs_e,
            "Rsh": rsh_e,
        }
        passes, reason = validate_electrical_curve(scalar_dict)
        if not passes and fault_class == "Healthy":
            # Reject only if healthy fails ranges
            logger.debug(f"Synthetic curve rejected: {reason}")
            return None

        gen_params = {
            "fault_class": fault_class,
            "irradiance": irradiance,
            "temperature": temperature,
            "I_L": float(I_L),
            "I_o": float(I_o),
            "R_s": float(R_s),
            "R_sh": float(R_sh),
            "a": float(a),
        }

        result = {
            "iv_curve": iv_curve,
            "pv_curve": pv_curve,
            "scalars": scalars,
            "generation_params": gen_params,
            "fault_class": fault_class,
            "source_conditioning_domain_id": "synth_elec",
        }

        if save:
            out_base = (Path(output_dir) / fault_class.replace("/", "_")) if output_dir else data_synthetic_electrical(fault_class)
            out_base.mkdir(parents=True, exist_ok=True)
            curve_id = str(uuid.uuid4())
            file_path = out_base / f"{curve_id}.npz"
            np.savez(
                str(file_path),
                iv_curve=iv_curve,
                pv_curve=pv_curve,
                scalars=scalars,
                generation_params=json.dumps(gen_params),
                source_conditioning_domain_id="synth_elec",
            )
            result["file_path"] = str(file_path)

        return result

    except Exception as e:
        logger.debug(f"Single diode simulation failed: {e}")
        return None


def generate_synthetic_electrical_dataset(
    n_samples_per_class: int = 15,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a full batch of synthetic electrical curves across all fault classes."""
    classes = [
        "Healthy",
        "Hotspot",
        "Open/Short-circuit",
        "Shading/Obstruction",
        "Soiling",
        "Structural (visual)",
        "Electrical/Diode fault",
    ]
    generated_count = 0
    attempt_count = 0
    results_by_class: dict[str, list[dict]] = {c: [] for c in classes}

    for fault_cls in classes:
        accepted = 0
        while accepted < n_samples_per_class and attempt_count < n_samples_per_class * 20:
            attempt_count += 1
            irr = float(np.random.uniform(600.0, 1050.0))
            temp = float(np.random.uniform(15.0, 50.0))
            res = generate_iv_curve_for_fault(
                fault_class=fault_cls,
                irradiance=irr,
                temperature=temp,
                output_dir=output_dir,
                save=True,
            )
            if res is not None:
                accepted += 1
                generated_count += 1
                results_by_class[fault_cls].append(res)

    acceptance_rate = generated_count / max(attempt_count, 1)
    logger.info(
        f"Synthetic Electrical Generation: {generated_count}/{attempt_count} accepted "
        f"({acceptance_rate * 100:.1f}% acceptance rate)"
    )
    return {
        "generated_count": generated_count,
        "attempt_count": attempt_count,
        "acceptance_rate": acceptance_rate,
        "results_by_class": results_by_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic electrical curves.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-samples", type=int, default=15, help="Samples per fault class")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else data_synthetic_electrical()
    summary = generate_synthetic_electrical_dataset(
        n_samples_per_class=args.n_samples,
        output_dir=out_dir,
    )
    print(f"Generated {summary['generated_count']} curves with acceptance rate {summary['acceptance_rate']*100:.1f}%.")
