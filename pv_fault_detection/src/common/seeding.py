"""
src/common/seeding.py
Deterministic seeding for reproducible experiments.

Seeds: Python random, numpy, torch (CPU+CUDA).
Controlled by config keys:
  seed: int
  training.deterministic: bool (slower but fully reproducible; default true for headline runs)
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value (from config key 'seed').
        deterministic: If True, sets torch.backends.cudnn.deterministic=True and
                       benchmark=False. Slower but fully reproducible. Default True
                       for headline runs; may be False for hyperparameter search speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can speed up training when input sizes are fixed
        torch.backends.cudnn.benchmark = True
