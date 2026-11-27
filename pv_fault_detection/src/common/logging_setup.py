"""
src/common/logging_setup.py
Structured logging and experiment tracker initialization.

Abstracts over MLflow / W&B / TensorBoard behind one interface.
The choice is a config value (logging.tracker), not a code branch.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard Python logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logger with a StreamHandler and optional FileHandler."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "run.log"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Experiment tracker abstraction
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Thin wrapper around MLflow / W&B / TensorBoard.

    Usage:
        tracker = ExperimentTracker.from_config(cfg)
        tracker.start_run(experiment_id, config)
        tracker.log_metric("macro_f1", 0.85, step=10)
        tracker.log_artifact("results/exp_id/confusion_matrix.png")
        tracker.end_run()
    """

    def __init__(self, backend: str, log_dir: str | Path) -> None:
        self.backend = backend
        self.log_dir = Path(log_dir)
        self._run = None

    @classmethod
    def from_config(cls, cfg: dict) -> "ExperimentTracker":
        backend = cfg.get("logging", {}).get("tracker", "mlflow")
        log_dir = cfg.get("logging", {}).get("log_dir", "logs/")
        return cls(backend=backend, log_dir=Path(log_dir))

    def start_run(self, experiment_id: str, config: dict) -> None:
        """Begin a tracked run, logging all config parameters."""
        if self.backend == "mlflow":
            self._start_mlflow(experiment_id, config)
        elif self.backend == "wandb":
            self._start_wandb(experiment_id, config)
        elif self.backend == "tensorboard":
            self._start_tensorboard(experiment_id)
        else:
            logger.warning(f"Unknown tracker backend '{self.backend}'. Logging disabled.")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self.backend == "mlflow":
            try:
                import mlflow
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"MLflow log_metric failed: {e}")
        elif self.backend == "wandb":
            try:
                import wandb
                wandb.log({key: value}, step=step)
            except Exception as e:
                logger.warning(f"W&B log_metric failed: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self.backend == "mlflow":
            try:
                import mlflow
                mlflow.log_artifact(str(path))
            except Exception as e:
                logger.warning(f"MLflow log_artifact failed: {e}")

    def end_run(self) -> None:
        if self.backend == "mlflow":
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow end_run failed: {e}")
        elif self.backend == "wandb":
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")

    # ------------------------------------------------------------------
    # Backend-specific private helpers
    # ------------------------------------------------------------------

    def _start_mlflow(self, experiment_id: str, config: dict) -> None:
        try:
            import os
            import mlflow
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"sqlite:///{(self.log_dir / 'mlflow.db').as_posix()}"
            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception:
                mlflow.set_tracking_uri((self.log_dir / "mlruns").as_uri())

            exp_name = config.get("experiment_name", experiment_id)
            mlflow.set_experiment(exp_name)
            mlflow.start_run(run_name=experiment_id)
            # Log all config params (flatten nested dict)
            flat_params = _flatten_dict(config)
            mlflow.log_params(flat_params)
        except ImportError:
            logger.warning("mlflow not installed. Run: pip install mlflow")
        except Exception as e:
            logger.warning(f"MLflow start_run failed: {e}")

    def _start_wandb(self, experiment_id: str, config: dict) -> None:
        try:
            import wandb
            wandb.init(
                project=config.get("experiment_name", "pv_fault_detection"),
                name=experiment_id,
                config=config,
                dir=str(self.log_dir),
            )
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    def _start_tensorboard(self, experiment_id: str) -> None:
        # TensorBoard writer is returned; metrics logged via log_metric won't work
        # without storing the writer. This is a minimal stub.
        logger.info(f"TensorBoard logging to {self.log_dir / 'tensorboard' / experiment_id}")


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dot-notation for MLflow param logging."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=key))
        else:
            # MLflow requires string values for params
            result[key] = str(v)
    return result
