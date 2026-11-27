"""
src/common/paths.py
Canonical path resolution. Never hard-code paths outside this module.

All path construction for the project goes through these helpers so that
the root can be relocated without hunting for hard-coded strings in src/.
"""
from __future__ import annotations

from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml or configs/base.yaml."""
    here = Path(__file__).resolve()
    for parent in [here.parent.parent.parent, here.parent.parent, here.parent]:
        if (parent / "pyproject.toml").exists() or (parent / "configs" / "base.yaml").exists():
            return parent
    # Fallback: cwd
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Directory accessors
# ---------------------------------------------------------------------------

def data_raw(dataset_id: str | None = None) -> Path:
    base = PROJECT_ROOT / "data" / "raw"
    return base / dataset_id if dataset_id else base


def data_processed_images(dataset_id: str | None = None) -> Path:
    base = PROJECT_ROOT / "data" / "processed" / "images"
    return base / dataset_id if dataset_id else base


def data_processed_electrical(dataset_id: str | None = None) -> Path:
    base = PROJECT_ROOT / "data" / "processed" / "electrical"
    return base / dataset_id if dataset_id else base


def data_processed_index() -> Path:
    return PROJECT_ROOT / "data" / "processed" / "index"


def data_synthetic_electrical(fault_class: str | None = None) -> Path:
    base = PROJECT_ROOT / "data" / "synthetic" / "electrical"
    if fault_class:
        safe_cls = fault_class.replace("/", "_")
        return base / safe_cls
    return base


def data_synthetic_images(modality: str | None = None, fault_class: str | None = None) -> Path:
    base = PROJECT_ROOT / "data" / "synthetic" / "images"
    if modality:
        base = base / modality
    if fault_class:
        safe_cls = fault_class.replace("/", "_")
        base = base / safe_cls
    return base


def dataset_status_md() -> Path:
    return PROJECT_ROOT / "data" / "DATASET_STATUS.md"


def results_dir(experiment_id: str | None = None) -> Path:
    base = PROJECT_ROOT / "results"
    return base / experiment_id if experiment_id else base


def checkpoints_dir(experiment_id: str | None = None) -> Path:
    base = PROJECT_ROOT / "checkpoints"
    return base / experiment_id if experiment_id else base


def logs_dir() -> Path:
    return PROJECT_ROOT / "logs"


def experiments_registry() -> Path:
    return PROJECT_ROOT / "experiments" / "registry.json"


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def label_maps_dir() -> Path:
    return PROJECT_ROOT / "src" / "data" / "label_maps"
