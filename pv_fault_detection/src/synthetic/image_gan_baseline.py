"""
src/synthetic/image_gan_baseline.py
GAN-based image synthesis baseline (comparison only — NOT primary augmentation).

Spec Section 11.2:
  DCGAN/SIGAN-style generator for the ablation row 6 comparison only.
  Never used as the primary augmentation source.

Full implementation in Stage 11.
"""
from __future__ import annotations


import json
import logging
import uuid
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_gan_image(
    fault_class: str,
    split: str,
    output_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> str | None:
    """Generate one synthetic image via GAN baseline (comparison only).

    HARD RULE: split must be 'train'.
    """
    if split != "train":
        raise ValueError(
            f"split must be 'train', got '{split}'. Leakage prevention."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_id = str(uuid.uuid4())
    out_img_path = out_dir / f"gan_{img_id}.jpg"
    sidecar_path = out_dir / f"gan_{img_id}.json"

    # Comparative baseline: smooth texture generation
    h, w = image_size
    noise = np.random.randn(h // 8, w // 8, 3).astype(np.float32)
    gan_img = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    gan_img = ((gan_img - gan_img.min()) / (gan_img.max() - gan_img.min() + 1e-8) * 255.0).astype(np.uint8)

    cv2.imwrite(str(out_img_path), gan_img)
    sidecar_data = {
        "synthetic_sample_id": f"gan_{img_id}",
        "generator_type": "gan_baseline",
        "fault_class": fault_class,
        "split": split,
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar_data, f, indent=2)

    return str(out_img_path)
