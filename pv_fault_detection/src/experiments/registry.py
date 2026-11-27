"""
src/experiments/registry.py
Experiment ID generation and append-only run registry.

Spec Section 16:
  experiment_id = f"{config_name}__{git_commit[:8]}__{seed}__{timestamp}"
  Appended to experiments/registry.json (append-only — never edited in place).
  Records git commit, git status (dirty-tree flag), config hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.common.paths import experiments_registry

logger = logging.getLogger(__name__)


def generate_experiment_id(config_name: str, seed: int) -> str:
    """Generate a unique experiment ID.

    Format: {config_name}__{git_commit[:8]}__{seed}__{timestamp_compact}
    """
    git_commit = _get_git_commit()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{config_name}__{git_commit}__{seed}__{ts}"


def register_run(
    experiment_id: str,
    cfg: dict,
    config_path: str | None = None,
) -> None:
    """Append run metadata to experiments/registry.json.

    Append-only — never overwrites existing entries.
    """
    registry_path = experiments_registry()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if registry_path.exists() and registry_path.stat().st_size > 2:
        with open(registry_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_name": cfg.get("experiment_name", "unnamed"),
        "seed": cfg.get("seed", 42),
        "git_commit": _get_git_commit(full=True),
        "git_dirty": _get_git_dirty(),
        "config_hash": _hash_config(cfg),
        "config_path": str(config_path) if config_path else None,
    }

    existing.append(entry)
    with open(registry_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Registered run: {experiment_id}")


def _get_git_commit(full: bool = False) -> str:
    """Get current git commit hash. Returns 'nogit' if git unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        commit = result.stdout.strip()
        return commit if full else commit[:8]
    except Exception:
        return "nogit"


def _get_git_dirty() -> bool:
    """Check if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _hash_config(cfg: dict) -> str:
    """Compute a short hash of the config dict for change detection."""
    cfg_str = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]
