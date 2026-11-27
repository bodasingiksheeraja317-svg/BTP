"""
src/models/domain_generalization/adversarial_dg.py
Adversarial domain generalization — gradient-reversal lambda scheduling.

Spec Section 12:
  Lambda schedule: linear warm-up over first 30% of training (standard DANN recipe).
  The actual gradient-reversal layer and domain head are in heads/domain_head.py.
  This module provides the schedule computation.
"""
from __future__ import annotations

import math


def compute_grl_lambda(
    current_step: int,
    total_steps: int,
    warmup_frac: float = 0.30,
    max_lambda: float = 1.0,
    schedule: str = "dann_linear_warmup",
) -> float:
    """Compute gradient-reversal lambda for the current training step.

    Args:
        current_step: Current global training step.
        total_steps: Total training steps.
        warmup_frac: Fraction of total steps for linear warm-up (0.30 per spec ASSUMPTION).
        max_lambda: Maximum lambda value after warm-up.
        schedule: 'dann_linear_warmup' (default) or 'dann_exponential'.

    Returns:
        Lambda value in [0, max_lambda].
    """
    if schedule == "dann_linear_warmup":
        warmup_steps = int(total_steps * warmup_frac)
        if current_step < warmup_steps:
            return max_lambda * (current_step / max(warmup_steps, 1))
        return max_lambda
    elif schedule == "dann_exponential":
        p = current_step / max(total_steps, 1)
        return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
    else:
        raise ValueError(f"Unknown GRL schedule: {schedule}")
