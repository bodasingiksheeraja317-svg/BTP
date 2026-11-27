"""
src/training/trainer.py
Single training loop — config-driven, used by EVERY experiment type.

Spec Section 13:
  One trainer, no "baseline trainer" vs "multimodal trainer" vs "DG trainer".
  DG method only changes what model returns and what losses.py adds,
  NEVER the loop structure itself.

Spec Section 13.2: total loss = fault_loss + coral_weight * coral_loss
                              + adversarial_weight * domain_loss
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Trainer:
    """Config-driven training loop for all experiment types.

    Usage:
        trainer = Trainer(model, cfg, tracker)
        results = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        tracker=None,
        experiment_id: str = "unnamed",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.tracker = tracker
        self.experiment_id = experiment_id

        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.loss_type = train_cfg.get("loss_type", "focal")
        self.focal_alpha = train_cfg.get("focal_alpha", 0.25)
        self.focal_gamma = train_cfg.get("focal_gamma", 2.0)
        self.coral_weight = train_cfg.get("coral_weight", 1.0)
        self.adversarial_weight = train_cfg.get("adversarial_weight", 1.0)
        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 10)
        self.dg_method = cfg.get("model", {}).get("dg_method", "none")
        self.mixed_precision = cfg.get("hardware", {}).get("mixed_precision", True)

        hw_cfg = cfg.get("hardware", {})
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and hw_cfg.get("device", "cuda") == "cuda"
            else "cpu"
        )
        self.model = self.model.to(self.device)

        from src.training.optim import build_optimizer
        self.optimizer = build_optimizer(model, cfg)
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and self.device.type == "cuda" else None

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> dict:
        """Run full training loop. Returns final metrics dict."""
        from src.training.losses import multilabel_focal_loss, class_weighted_bce_loss
        from src.training.optim import build_scheduler

        total_steps = self.epochs * max(len(train_loader), 1)
        self.total_steps = total_steps
        self.global_step = 0
        scheduler = build_scheduler(self.optimizer, self.cfg, total_steps=total_steps)

        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_macro_f1": []}

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader, scheduler)
            val_metrics = self._val_epoch(val_loader)
            val_f1 = val_metrics.get("macro_f1", 0.0)

            history["train_loss"].append(train_loss)
            history["val_macro_f1"].append(val_f1)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"train_loss={train_loss:.4f}, val_macro_f1={val_f1:.4f}"
            )

            if self.tracker:
                self.tracker.log_metrics(
                    {"train_loss": train_loss, "val_macro_f1": val_f1},
                    step=epoch,
                )

            # Checkpoint
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            self._save_checkpoint(epoch, val_metrics, is_best=False)

        return {"best_val_macro_f1": best_val_f1, "history": history}

    def _train_epoch(self, loader, scheduler) -> float:
        """One training epoch. Returns mean loss."""
        from src.training.losses import multilabel_focal_loss, class_weighted_bce_loss
        from src.models.domain_generalization.coral import coral_loss
        from src.models.domain_generalization.adversarial_dg import compute_grl_lambda

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            self.optimizer.zero_grad()

            if self.dg_method == "adversarial":
                grl_lambda = compute_grl_lambda(self.global_step, max(self.total_steps, 1))
                if hasattr(self.model, "domain_head") and hasattr(self.model.domain_head, "set_alpha"):
                    self.model.domain_head.set_alpha(grl_lambda)
                if self.global_step % 20 == 0:
                    logger.debug(f"[GRL] Step {self.global_step}/{self.total_steps} alpha={grl_lambda:.4f}")

            device_type = self.device.type
            with torch.amp.autocast(device_type=device_type, enabled=self.scaler is not None and device_type == "cuda"):
                outputs = self.model(batch)
                fault_logits = outputs["fault_logits"]
                z = outputs["z"]
                domain_logits = outputs.get("domain_logits")

                targets = batch.get("fault_targets").to(self.device).float()

                if self.loss_type == "focal":
                    fault_loss = multilabel_focal_loss(
                        fault_logits, targets,
                        alpha=self.focal_alpha, gamma=self.focal_gamma
                    )
                else:
                    fault_loss = class_weighted_bce_loss(fault_logits, targets)

                dg_loss = torch.tensor(0.0, device=self.device)
                if self.dg_method == "coral":
                    domain_ids = batch.get("domain_id", batch.get("domain_ids", []))
                    if domain_ids:
                        from collections import defaultdict
                        domain_feats = defaultdict(list)
                        for i, did in enumerate(domain_ids):
                            domain_feats[did].append(z[i])
                        source_feats = [torch.stack(v) for v in domain_feats.values() if len(v) > 1]
                        if len(source_feats) >= 2:
                            dg_loss = coral_loss(source_feats) * self.coral_weight

                elif self.dg_method == "adversarial" and domain_logits is not None:
                    from src.training.losses import adversarial_domain_loss
                    domain_targets = batch.get("domain_targets")
                    if domain_targets is not None:
                        dg_loss = adversarial_domain_loss(domain_logits, domain_targets.to(self.device)) * self.adversarial_weight

                loss = fault_loss + dg_loss

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, loader) -> dict:
        """One validation epoch. Returns metrics dict."""
        self.model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(batch)
                all_logits.append(outputs["fault_logits"].cpu())
                targets = batch.get("fault_targets")
                if targets is not None:
                    all_targets.append(targets.cpu())

        if not all_logits or not all_targets:
            return {"macro_f1": 0.0}

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        preds = (torch.sigmoid(logits) > 0.5).float()

        from sklearn.metrics import f1_score
        f1 = f1_score(
            targets.numpy(), preds.numpy(), average="macro", zero_division=0
        )
        return {"macro_f1": float(f1)}

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool
    ) -> None:
        """Save model checkpoint."""
        from src.common.paths import checkpoints_dir
        ckpt_dir = checkpoints_dir(self.experiment_id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "experiment_id": self.experiment_id,
        }

        suffix = "best" if is_best else "last"
        ckpt_path = ckpt_dir / f"checkpoint_{suffix}.pt"
        torch.save(state, ckpt_path)

        meta = {"epoch": epoch, "metrics": metrics, "experiment_id": self.experiment_id}
        meta_path = ckpt_dir / f"checkpoint_{suffix}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
