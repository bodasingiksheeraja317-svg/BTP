import json
import logging
import uuid
from pathlib import Path
import cv2
import numpy as np

from src.synthetic.quality_control import validate_synthetic_image_ssim

logger = logging.getLogger(__name__)

# ============================================================
# IMPORTANT: Despite the module name, this does NOT implement
# DDPM or Stable Diffusion-based inpainting.
#
# Spec Section 11.2 calls for diffusers-based DDPM inpainting,
# but that requires CUDA + ~4GB model download. In this CPU-only
# environment, we implement physics-motivated geometric augmentation
# (Gaussian thermal spots, shadow masks, crack lines) with SSIM
# quality filtering.
#
# This is an honest deviation from the spec, documented in STATUS.md.
# If CUDA + diffusers become available, replace the body of
# generate_synthetic_image() with actual diffusers DDPM inpainting.
# The function signature and sidecar JSON format should remain the same.
# ============================================================


def generate_synthetic_image(
    source_sample_id: str,
    source_image_path: str,
    fault_class: str,
    split: str,
    output_dir: str | Path,
    ssim_threshold: float = 0.70,
) -> str | None:
    """Generate one synthetic fault image via geometry-based defect augmentation.

    NOTE: This uses physics-motivated geometric augmentation (Gaussian hotspot
    masks, shadow polygons, crack lines), NOT DDPM/Stable Diffusion.
    See module header for explanation.

    HARD RULE: split must be 'train'. Raises ValueError otherwise.
    (Prevents conditioning on test-split images — spec Section 11.4 leakage rule.)

    Returns file path of generated image if accepted by SSIM quality filter, else None.
    """
    if split != "train":
        raise ValueError(
            f"Synthetic image generation requires split='train', got '{split}'. "
            "This is a leakage prevention rule — never generate from test-split images."
        )

    img_p = Path(source_image_path)
    if not img_p.exists():
        logger.warning(f"Source image not found: {source_image_path}")
        return None

    img = cv2.imread(str(img_p))
    if img is None:
        return None

    h, w = img.shape[:2]
    defect_img = img.copy().astype(np.float32)

    # Physics-based inpainting defect simulation per fault class
    if fault_class == "Hotspot":
        # Localized intense thermal heating spot
        cx, cy = np.random.randint(int(w * 0.2), int(w * 0.8)), np.random.randint(int(h * 0.2), int(h * 0.8))
        rx, ry = np.random.randint(int(w * 0.05), int(w * 0.15)), np.random.randint(int(h * 0.05), int(h * 0.15))
        y, x = np.ogrid[:h, :w]
        mask = np.exp(-(((x - cx)**2) / (2 * rx**2) + ((y - cy)**2) / (2 * ry**2)))
        intensity = np.random.uniform(40.0, 100.0)
        for c in range(defect_img.shape[2]):
            defect_img[:, :, c] += mask * intensity

    elif fault_class in ("Shading/Obstruction", "Shading_Obstruction"):
        # Polygon shadow mask
        pts = np.array([
            [np.random.randint(0, int(w * 0.4)), np.random.randint(0, int(h * 0.5))],
            [np.random.randint(int(w * 0.6), w), np.random.randint(0, int(h * 0.5))],
            [np.random.randint(int(w * 0.4), w), np.random.randint(int(h * 0.5), h)],
            [np.random.randint(0, int(w * 0.4)), np.random.randint(int(h * 0.5), h)],
        ], np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask_f = (mask / 255.0)[:, :, np.newaxis]
        attenuation = np.random.uniform(0.3, 0.6)
        defect_img = defect_img * (1.0 - mask_f * (1.0 - attenuation))

    elif fault_class == "Soiling":
        # Diffuse attenuation with surface gradient
        grad = np.linspace(0.85, 0.95, h)[:, np.newaxis, np.newaxis]
        noise = np.random.normal(0, 5.0, img.shape)
        defect_img = defect_img * grad + noise

    elif fault_class in ("Structural (visual)", "Structural"):
        # Crack / fracture defect line
        pt1 = (np.random.randint(int(w * 0.2), int(w * 0.8)), np.random.randint(int(h * 0.1), int(h * 0.4)))
        pt2 = (np.random.randint(int(w * 0.2), int(w * 0.8)), np.random.randint(int(h * 0.6), int(h * 0.9)))
        crack_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(crack_mask, pt1, pt2, 255, thickness=np.random.randint(2, 5))
        crack_f = (crack_mask / 255.0)[:, :, np.newaxis]
        defect_img = defect_img * (1.0 - crack_f) + crack_f * 20.0

    elif fault_class in ("Electrical/Diode fault", "Electrical_Diode fault", "Open/Short-circuit", "Open_Short-circuit"):
        # Column / string heating stripe
        x_start = np.random.randint(int(w * 0.1), int(w * 0.7))
        x_end = x_start + np.random.randint(int(w * 0.1), int(w * 0.3))
        stripe_mask = np.zeros((h, w, 1), dtype=np.float32)
        stripe_mask[:, x_start:x_end, :] = 1.0
        delta = np.random.uniform(30.0, 70.0)
        defect_img += stripe_mask * delta

    defect_img = np.clip(defect_img, 0, 255).astype(np.uint8)

    # Validate with SSIM quality control filter
    passes_ssim, ssim_val = validate_synthetic_image_ssim(defect_img, img, ssim_threshold=ssim_threshold)
    if not passes_ssim:
        logger.debug(f"Synthetic image rejected by SSIM: {ssim_val:.3f} < {ssim_threshold}")
        return None

    # Save image and sidecar metadata
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_id = str(uuid.uuid4())
    out_img_path = out_dir / f"{img_id}.jpg"
    sidecar_path = out_dir / f"{img_id}.json"

    cv2.imwrite(str(out_img_path), defect_img)
    sidecar_data = {
        "synthetic_sample_id": f"synth_img_{img_id}",
        "source_real_sample_id": source_sample_id,
        "source_image_path": str(source_image_path),
        "fault_class": fault_class,
        "split": split,
        "ssim": float(ssim_val),
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar_data, f, indent=2)

    return str(out_img_path)
