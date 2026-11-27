"""
tests/leakage/test_synthetic_not_from_test_split.py
Leakage test: every synthetic sample's source is guaranteed to come from the training split.

This is a LOAD-BEARING test. If it fails, the synthetic generation/mixing pipeline is wrong.
"""
from pathlib import Path
import pytest
from src.synthetic.image_diffusion import generate_synthetic_image
from src.synthetic.image_gan_baseline import generate_gan_image
from src.synthetic.mixing import SyntheticMixer

pytestmark = [pytest.mark.leakage]


def test_synthetic_mixer_rejects_non_train_splits():
    """SyntheticMixer must raise ValueError if created for val or test splits."""
    for split in ["val", "test", "evaluation", "test_set"]:
        with pytest.raises(ValueError) as excinfo:
            SyntheticMixer(ratio=0.5, split=split)
        assert "requires split='train'" in str(excinfo.value)


def test_image_diffusion_rejects_non_train_splits():
    """generate_synthetic_image must raise ValueError if called with split != 'train'."""
    for split in ["val", "test", "eval"]:
        with pytest.raises(ValueError) as excinfo:
            generate_synthetic_image(
                source_sample_id="img_1",
                source_image_path="dummy.jpg",
                fault_class="Hotspot",
                split=split,
                output_dir="out",
            )
        assert "requires split='train'" in str(excinfo.value)


def test_image_gan_rejects_non_train_splits():
    """generate_gan_image must raise ValueError if called with split != 'train'."""
    for split in ["val", "test", "eval"]:
        with pytest.raises(ValueError) as excinfo:
            generate_gan_image(
                fault_class="Hotspot",
                split=split,
                output_dir="out",
            )
        assert "split must be 'train'" in str(excinfo.value)


def test_synthetic_image_train_path_and_sidecar_tracking(tmp_path):
    """Verify generate_synthetic_image on split='train' produces valid image and sidecar provenance metadata."""
    import json
    import cv2
    import numpy as np

    # 1. Create a dummy real training image
    train_img_path = tmp_path / "real_train_sample.jpg"
    dummy_img = (np.random.rand(224, 224, 3) * 200 + 30).astype(np.uint8)
    cv2.imwrite(str(train_img_path), dummy_img)

    synth_out_dir = tmp_path / "synthetic_out"
    gen_path = generate_synthetic_image(
        source_sample_id="real_train_sample_123",
        source_image_path=str(train_img_path),
        fault_class="Hotspot",
        split="train",
        output_dir=synth_out_dir,
        ssim_threshold=0.60,
    )

    assert gen_path is not None, "generate_synthetic_image should generate valid image"
    assert Path(gen_path).exists()

    # Verify sidecar JSON tracking
    json_path = Path(gen_path).with_suffix(".json")
    assert json_path.exists(), "Sidecar metadata JSON must exist alongside synthetic image"

    with open(json_path) as f:
        sidecar = json.load(f)

    assert sidecar["source_real_sample_id"] == "real_train_sample_123"
    assert sidecar["split"] == "train"
    assert sidecar["fault_class"] == "Hotspot"
    assert sidecar["ssim"] >= 0.60


def test_synthetic_mixer_mixing_proportions(tmp_path):
    """Verify SyntheticMixer mixes real and synthetic datasets according to configured ratio."""
    import pandas as pd

    real_df = pd.DataFrame([{"id": f"real_{i}"} for i in range(100)])
    synth_df = pd.DataFrame([{"id": f"synth_{i}"} for i in range(100)])

    # Test ratio 0.5 -> 100 real + 50 synth = 150
    mixer = SyntheticMixer(ratio=0.5, split="train")
    mixed_df = mixer.mix(real_df, synth_df)
    assert len(mixed_df) == 150
    assert len(mixed_df[mixed_df["id"].str.startswith("real_")]) == 100
    assert len(mixed_df[mixed_df["id"].str.startswith("synth_")]) == 50

    # Test ratio 0.0 -> 100 real
    mixer_zero = SyntheticMixer(ratio=0.0, split="train")
    assert len(mixer_zero.mix(real_df, synth_df)) == 100
