"""
src/synthetic/mixing.py
Real/synthetic data mixing for training.

Spec Section 11.4:
  HARD RULE enforced in code: generate() requires split='train',
  raises ValueError if called with split != 'train'.
  
  Supported ratios (config synthetic.ratio):
    0.0: real-only
    0.1, 0.25, 0.5, 1.0: mixed at ratio
    'unlimited': synthetic-only
    'pretrain_finetune': two-stage (synthetic first, then real fine-tune)
    'domain_randomized': synthetic with swept conditioning parameters

  mixing.py never computes metrics — only produces a mixed dataloader.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SyntheticMixer:
    """Controls real/synthetic data mixing for training splits."""

    VALID_RATIOS = {0.0, 0.1, 0.25, 0.5, 1.0, "unlimited", "pretrain_finetune", "domain_randomized"}

    def __init__(self, ratio, split: str) -> None:
        if split != "train":
            raise ValueError(
                f"SyntheticMixer requires split='train', got '{split}'. "
                "Synthetic data must only be mixed into training splits. "
                "This is a hard leakage-prevention rule (spec Section 11.4)."
            )
        if ratio not in self.VALID_RATIOS:
            raise ValueError(
                f"Invalid synthetic ratio '{ratio}'. Allowed: {self.VALID_RATIOS}"
            )
        self.ratio = ratio
        self.split = split

    def mix(self, real_samples, synthetic_samples):
        """Produce a mixed sample list or DataFrame according to configured synthetic.ratio."""
        import pandas as pd

        if self.ratio == 0.0 or synthetic_samples is None or len(synthetic_samples) == 0:
            return real_samples

        if self.ratio == "unlimited":
            return synthetic_samples

        if self.ratio == "pretrain_finetune":
            return {"pretrain": synthetic_samples, "finetune": real_samples}

        if isinstance(real_samples, pd.DataFrame):
            n_synth = int(len(real_samples) * float(self.ratio)) if isinstance(self.ratio, (int, float)) else len(synthetic_samples)
            if isinstance(synthetic_samples, pd.DataFrame):
                n_take = min(n_synth, len(synthetic_samples))
                synth_sub = synthetic_samples.iloc[:n_take]
                return pd.concat([real_samples, synth_sub], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
            return real_samples

        # List of items
        n_synth = int(len(real_samples) * float(self.ratio)) if isinstance(self.ratio, (int, float)) else len(synthetic_samples)
        n_take = min(n_synth, len(synthetic_samples))
        mixed = list(real_samples) + list(synthetic_samples[:n_take])
        return mixed
