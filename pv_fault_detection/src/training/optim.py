"""
src/training/optim.py
Optimizer and LR scheduler construction from config.

Spec Section 13.3:
  - AdamW default (ASSUMPTION, config-overridable)
  - Cosine LR schedule with linear warmup
  - Config keys: training.optimizer, training.lr, training.weight_decay, training.warmup_steps
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    train_cfg = cfg.get("training", {})
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-5)

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = train_cfg.get("momentum", 0.9)
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Supported: adamw, sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    total_steps: int,
) -> LRScheduler:
    """Build cosine LR scheduler with linear warmup."""
    train_cfg = cfg.get("training", {})
    warmup_steps = train_cfg.get("warmup_steps", 500)

    return CosineWithWarmup(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)


class CosineWithWarmup(LRScheduler):
    """Cosine LR schedule with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]
