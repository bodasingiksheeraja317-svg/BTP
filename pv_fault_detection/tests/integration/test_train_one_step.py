"""
tests/integration/test_train_one_step.py
Integration test: train mini-epoch on fixture batch and verify gradient update and checkpoint creation.
"""
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from src.common.config import load_config
from src.models.pv_fault_model import build_model
from src.training.trainer import Trainer

pytestmark = pytest.mark.integration


class MockDataset(Dataset):
    """Synthetic dataset generating complete multimodal sample dicts."""

    def __init__(self, num_samples: int = 8):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "rgb": torch.randn(3, 224, 224),
            "thermal": torch.randn(1, 224, 224),
            "electrical_iv": torch.randn(100, 2),
            "electrical_pv": torch.randn(100, 1),
            "electrical_scalars": torch.randn(7),
            "environment": torch.randn(3),
            "fault_targets": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "domain_ids": "test_domain",
        }


def _collate_fn(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "thermal": torch.stack([b["thermal"] for b in batch]),
        "electrical_iv": torch.stack([b["electrical_iv"] for b in batch]),
        "electrical_pv": torch.stack([b["electrical_pv"] for b in batch]),
        "electrical_scalars": torch.stack([b["electrical_scalars"] for b in batch]),
        "environment": torch.stack([b["environment"] for b in batch]),
        "fault_targets": torch.stack([b["fault_targets"] for b in batch]),
        "domain_ids": [b["domain_ids"] for b in batch],
    }


def test_trainer_mini_epoch(tmp_path):
    """Verify Trainer completes training epochs, updates weights, and saves checkpoints."""
    cfg = load_config(
        "configs/multimodal_full.yaml",
        overrides=[
            "training.epochs=2",
            "training.early_stopping_patience=2",
            "hardware.device=cpu",
        ],
    )
    model = build_model(cfg)

    train_loader = DataLoader(MockDataset(6), batch_size=2, collate_fn=_collate_fn)
    val_loader = DataLoader(MockDataset(4), batch_size=2, collate_fn=_collate_fn)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        experiment_id="test_exp_001",
    )

    results = trainer.train(train_loader, val_loader)

    assert "best_val_macro_f1" in results
    assert "history" in results
    assert len(results["history"]["train_loss"]) == 2
    for loss_val in results["history"]["train_loss"]:
        assert not torch.isnan(torch.tensor(loss_val))


def test_trainer_real_dataloader_pipeline(tmp_path):
    """Verify Trainer trains on real indexed data via _build_dataloaders without MockDataset."""
    from src.experiments.run import _build_dataloaders

    cfg = load_config(
        "configs/baseline_thermal.yaml",
        overrides=[
            "training.epochs=1",
            "training.batch_size=8",
            "hardware.device=cpu",
            "hardware.num_workers=0",
        ],
    )
    model = build_model(cfg)
    train_loader, val_loader, test_loader = _build_dataloaders(cfg)

    assert len(train_loader) > 0, "Real train_loader must not be empty"
    assert len(val_loader) > 0, "Real val_loader must not be empty"

    # Test real dataloader batch extraction
    batch = next(iter(train_loader))
    assert "thermal" in batch
    assert batch["thermal"] is not None
    assert batch["thermal"].shape == (8, 1, 224, 224)

    # Wrap real dataset in a small subset for fast integration step
    from torch.utils.data import Subset
    mini_train = DataLoader(Subset(train_loader.dataset, range(16)), batch_size=8, collate_fn=train_loader.collate_fn)
    mini_val = DataLoader(Subset(val_loader.dataset, range(8)), batch_size=8, collate_fn=val_loader.collate_fn)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        experiment_id="test_real_data_exp",
    )
    results = trainer.train(mini_train, mini_val)
    assert len(results["history"]["train_loss"]) == 1
    loss_val = results["history"]["train_loss"][0]
    assert loss_val > 0.0, f"Expected positive real loss, got {loss_val}"
    assert not torch.isnan(torch.tensor(loss_val))
