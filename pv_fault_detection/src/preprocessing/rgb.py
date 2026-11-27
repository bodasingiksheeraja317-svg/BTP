"""
src/preprocessing/rgb.py
RGB image preprocessing pipeline.

Pure-function library — no side effects, no YAML reading.
Parameters are passed in from caller (dataloader construction) which reads config once.
This keeps preprocessing unit-testable without a config file.

Spec reference: Section 9.3
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default normalization (ImageNet statistics)
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_and_preprocess(
    image_path: str | Path,
    image_size: int = 224,
    normalize_mean: tuple = _DEFAULT_MEAN,
    normalize_std: tuple = _DEFAULT_STD,
    blur_filter_threshold: float = 100.0,
    augment: bool = False,
) -> np.ndarray | None:
    """Load and preprocess a single RGB image.

    Args:
        image_path: Path to the image file.
        image_size: Target square size (224 per spec Section 10).
        normalize_mean: Per-channel mean for normalization.
        normalize_std: Per-channel std for normalization.
        blur_filter_threshold: Reject images with Laplacian variance below this.
                               Returns None for rejected images.
        augment: Apply albumentations augmentations if True (training only).

    Returns:
        Preprocessed image as float32 ndarray [C, H, W] normalized to [0,1] then standardized,
        or None if image fails quality check.
    """
    import cv2

    img_path = Path(image_path)
    if not img_path.exists():
        logger.warning(f"RGB image not found: {img_path}")
        return None

    # Load
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Failed to load RGB image: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Blur filter (Laplacian variance)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_filter_threshold:
        logger.debug(f"Image rejected (blur score {blur_score:.1f} < {blur_filter_threshold}): {img_path}")
        return None

    # Resize
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Augmentation (albumentations, training only)
    if augment:
        img = _apply_augmentations(img)

    # Normalize: [0,255] -> [0,1] -> standardize
    img = img.astype(np.float32) / 255.0
    mean = np.array(normalize_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(normalize_std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    return img


def _apply_augmentations(img: np.ndarray) -> np.ndarray:
    """Apply albumentations augmentations for training."""
    try:
        import albumentations as A
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        ])
        result = transform(image=img)
        return result["image"]
    except ImportError:
        logger.warning("albumentations not installed; skipping augmentation")
        return img
