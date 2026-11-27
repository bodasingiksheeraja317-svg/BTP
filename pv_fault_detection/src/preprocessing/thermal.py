"""
src/preprocessing/thermal.py
Thermal image preprocessing pipeline.

Handles emissivity/reflected-temperature correction and
radiometric channel normalization. Pure-function library.

Spec reference: Section 9.3
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_and_preprocess(
    thermal_path: str | Path,
    thermal_raw_path: str | Path | None = None,
    image_size: int = 224,
    use_raw_radiometric: bool = False,
    emissivity: float = 0.95,
    reflected_temp_c: float = 25.0,
    augment: bool = False,
) -> np.ndarray | None:
    """Load and preprocess a thermal image.

    Args:
        thermal_path: Path to the thermal image (JPEG/PNG falsecolor or grayscale).
        thermal_raw_path: Optional .npy raw radiometric temperature array.
        image_size: Target square size.
        use_raw_radiometric: If True and thermal_raw_path is provided, add raw
                             temperature as a second channel.
        emissivity: Panel emissivity for radiometric correction (typical Si panel ~0.95).
        reflected_temp_c: Reflected apparent temperature in Celsius.
        augment: Apply augmentations if True.

    Returns:
        Preprocessed thermal image as float32 ndarray:
          - 1-channel [1, H, W] if use_raw_radiometric=False or no raw path
          - 2-channel [2, H, W] if use_raw_radiometric=True and raw path exists
        Returns None if load fails.
    """
    import cv2

    img_path = Path(thermal_path)
    if not img_path.exists():
        logger.warning(f"Thermal image not found: {img_path}")
        return None

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Failed to load thermal image: {img_path}")
        return None

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)

    # Normalize grayscale channel to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    channels = [img[np.newaxis, :, :]]  # [1, H, W]

    # Add raw radiometric channel if available and requested
    if use_raw_radiometric and thermal_raw_path is not None:
        raw_path = Path(thermal_raw_path)
        if raw_path.exists():
            raw_temp = np.load(str(raw_path)).astype(np.float32)
            raw_temp = cv2.resize(raw_temp, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            # Emissivity-corrected temperature
            raw_temp = _apply_emissivity_correction(raw_temp, emissivity, reflected_temp_c)
            # Normalize raw temperature channel
            t_min, t_max = raw_temp.min(), raw_temp.max()
            if t_max > t_min:
                raw_temp = (raw_temp - t_min) / (t_max - t_min)
            channels.append(raw_temp[np.newaxis, :, :])

    if augment:
        channels = _apply_augmentations(channels)

    return np.concatenate(channels, axis=0)  # [1 or 2, H, W]


def _apply_emissivity_correction(
    raw_temp: np.ndarray, emissivity: float, reflected_temp_c: float
) -> np.ndarray:
    """Apply simple emissivity correction to raw radiometric temperature.

    T_object = (T_measured - (1 - emissivity) * T_reflected) / emissivity
    """
    return (raw_temp - (1.0 - emissivity) * reflected_temp_c) / emissivity


def _apply_augmentations(channels: list) -> list:
    """Apply augmentations to thermal channels."""
    try:
        import albumentations as A
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        # Apply to first channel image only (augmentations are spatial, same for all channels)
        img_hw = channels[0][0]  # [H, W]
        result = transform(image=img_hw)
        augmented_hw = result["image"]
        channels[0] = augmented_hw[np.newaxis, :, :]
    except ImportError:
        pass
    return channels
