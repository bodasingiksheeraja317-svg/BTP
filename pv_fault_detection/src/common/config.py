"""
src/common/config.py
Configuration loading, hierarchical merging, and cross-key validation.

Design decisions (spec Section 18):
- Simple YAML merge: base -> experiment file -> CLI overrides
- NOT Hydra (chosen for simplicity per spec)
- Validates cross-key consistency at load time (fail fast)
- Key rule enforced here: dg_method != "none" requires protocol == "lodo"
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load and return the fully-resolved config.

    Resolution order (later wins):
      1. configs/base.yaml  (always loaded)
      2. config_path        (experiment-specific overrides)
      3. overrides          (CLI --override key=value pairs)

    Args:
        config_path: Path to the experiment YAML file.
        overrides:   List of "key.subkey=value" strings from CLI.

    Returns:
        Fully-merged, validated config dict.

    Raises:
        ConfigValidationError: if cross-key consistency checks fail.
        FileNotFoundError: if base.yaml or config_path don't exist.
    """
    base_path = _find_base_config()
    base_cfg = _load_yaml(base_path)
    exp_cfg = _load_yaml(config_path)
    merged = _deep_merge(base_cfg, exp_cfg)

    if overrides:
        for override in overrides:
            merged = _apply_override(merged, override)

    _validate(merged)
    return merged


def save_config(cfg: dict, output_path: str | Path) -> None:
    """Dump the fully-resolved config to YAML (for experiment snapshotting)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class ConfigValidationError(ValueError):
    """Raised when cross-key consistency checks fail."""


def _find_base_config() -> Path:
    """Locate configs/base.yaml relative to the project root."""
    # Walk up from this file's location until we find configs/base.yaml
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "configs" / "base.yaml"
        if candidate.exists():
            return candidate
    # Fallback: look relative to cwd
    candidate = Path("configs") / "base.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not find configs/base.yaml. Run from the project root directory."
    )


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _apply_override(cfg: dict, override: str) -> dict:
    """Apply a single "dotted.key=value" CLI override to the config dict.

    Supports nested keys via dot notation: "model.fusion_type=late"
    Values are parsed as YAML scalars (handles int, float, bool, str, null).
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid override format '{override}'. Expected 'key.subkey=value'."
        )
    key_path, raw_value = override.split("=", 1)
    value = yaml.safe_load(raw_value)  # parse type correctly

    keys = key_path.strip().split(".")
    target = cfg
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value
    return cfg


def _validate(cfg: dict) -> None:
    """Enforce cross-key consistency rules. Raises ConfigValidationError on violation."""
    dg_method = cfg.get("model", {}).get("dg_method", "none")
    protocol = cfg.get("evaluation", {}).get("protocol", "random_split")

    # Rule: DG method requires LODO protocol (spec Section 12)
    if dg_method != "none" and protocol != "lodo":
        raise ConfigValidationError(
            f"model.dg_method='{dg_method}' requires evaluation.protocol='lodo', "
            f"but got evaluation.protocol='{protocol}'. "
            "DG methods have no meaning under random/in-domain splits. "
            "Set evaluation.protocol=lodo in your config or via --override."
        )

    # Rule: synthetic ratio must be a known value
    valid_ratios = {0.0, 0.1, 0.25, 0.5, 1.0, "unlimited", "pretrain_finetune", "domain_randomized"}
    ratio = cfg.get("synthetic", {}).get("ratio", 0.0)
    if ratio not in valid_ratios:
        raise ConfigValidationError(
            f"synthetic.ratio='{ratio}' is not a recognized value. "
            f"Allowed values: {sorted(str(v) for v in valid_ratios)}"
        )

    # Rule: modalities must be a non-empty subset of [rgb, thermal, electrical]
    valid_modalities = {"rgb", "thermal", "electrical"}
    modalities = cfg.get("model", {}).get("modalities", [])
    if not modalities:
        raise ConfigValidationError("model.modalities must not be empty.")
    unknown = set(modalities) - valid_modalities
    if unknown:
        raise ConfigValidationError(
            f"Unknown modalities in model.modalities: {unknown}. "
            f"Allowed: {valid_modalities}"
        )
