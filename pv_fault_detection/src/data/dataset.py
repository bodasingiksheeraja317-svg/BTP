"""
src/data/dataset.py
PyTorch Dataset implementation that reads canonical Parquet index and loads preprocessed tensors.

Handles:
- Missing modalities (returns None for absent modalities, converted to proper batch tensors by collator)
- Multi-label targets (encoded to [7] binary float tensor)
- Environmental features and known flags
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.paths import PROJECT_ROOT
from src.data.adapters.base import CANONICAL_FAULT_CLASSES
from src.preprocessing.electrical import load_electrical_record, preprocess_iv_curve, derive_pv_curve, extract_scalar_features
from src.preprocessing.environment import process_environment
from src.preprocessing.rgb import load_and_preprocess as preprocess_rgb
from src.preprocessing.thermal import load_and_preprocess as preprocess_thermal

logger = logging.getLogger(__name__)


class PVDataset(Dataset):
    """Dataset for canonical multimodal PV samples."""

    def __init__(
        self,
        df: pd.DataFrame,
        modalities: list[str] = ["rgb", "thermal", "electrical"],
        is_training: bool = False,
        image_size: int = 224,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.modalities = set(modalities)
        self.is_training = is_training
        self.image_size = image_size
        self.class_to_idx = {cls: i for i, cls in enumerate(CANONICAL_FAULT_CLASSES)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # 1. Fault labels -> multi-label binary tensor [7]
        fault_labels_raw = row["fault_label"]
        if isinstance(fault_labels_raw, str):
            try:
                fault_labels = json.loads(fault_labels_raw.replace("'", '"'))
            except Exception:
                fault_labels = [fault_labels_raw]
        else:
            fault_labels = fault_labels_raw

        target = torch.zeros(len(CANONICAL_FAULT_CLASSES), dtype=torch.float32)
        for lbl in fault_labels:
            if lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1.0

        sample: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "domain_id": row["domain_id"],
            "site_id": row["site_id"],
            "fault_targets": target,
            "rgb": None,
            "thermal": None,
            "electrical_iv": None,
            "electrical_pv": None,
            "electrical_scalars": None,
            "environment": None,
            "scalar_known_mask": None,
            "env_known_mask": None,
        }

        # 2. RGB image
        if "rgb" in self.modalities and pd.notna(row.get("rgb_path")) and row.get("rgb_path"):
            rgb_file = PROJECT_ROOT / str(row["rgb_path"])
            rgb_arr = preprocess_rgb(rgb_file, image_size=self.image_size, augment=self.is_training)
            if rgb_arr is not None:
                sample["rgb"] = torch.from_numpy(rgb_arr).float()

        # 3. Thermal image
        if "thermal" in self.modalities and pd.notna(row.get("thermal_path")) and row.get("thermal_path"):
            thermal_file = PROJECT_ROOT / str(row["thermal_path"])
            raw_path = PROJECT_ROOT / str(row["thermal_raw_path"]) if pd.notna(row.get("thermal_raw_path")) and row.get("thermal_raw_path") else None
            thermal_arr = preprocess_thermal(
                thermal_file,
                thermal_raw_path=raw_path,
                image_size=self.image_size,
                augment=self.is_training,
            )
            if thermal_arr is not None:
                sample["thermal"] = torch.from_numpy(thermal_arr).float()

        # 4. Electrical
        if "electrical" in self.modalities and pd.notna(row.get("electrical_path")) and row.get("electrical_path"):
            elec_file = PROJECT_ROOT / str(row["electrical_path"])
            elec_rec = load_electrical_record(elec_file)
            if elec_rec is not None:
                iv = preprocess_iv_curve(elec_rec["iv_curve"])
                pv = derive_pv_curve(iv)
                scalars = extract_scalar_features(iv)
                sample["electrical_iv"] = torch.from_numpy(iv).float()
                sample["electrical_pv"] = torch.from_numpy(pv).float()
                sample["electrical_scalars"] = torch.from_numpy(scalars).float()
                sample["scalar_known_mask"] = torch.ones(7, dtype=torch.bool)

        # 5. Environment
        env_raw = row.get("environment")
        if pd.notna(env_raw) and env_raw:
            if isinstance(env_raw, str):
                try:
                    env_dict = json.loads(env_raw)
                except Exception:
                    env_dict = None
            else:
                env_dict = env_raw
        else:
            env_dict = None

        env_vals, env_mask = process_environment(env_dict)
        sample["environment"] = torch.from_numpy(env_vals).float()
        sample["env_known_mask"] = torch.from_numpy(env_mask)

        return sample


def pv_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates list of sample dicts into a batch dict for PVFaultModel with per-sample modality masks."""
    out: dict[str, Any] = {}

    for key in ["sample_id", "domain_id", "site_id"]:
        out[key] = [b[key] for b in batch]

    out["fault_targets"] = torch.stack([b["fault_targets"] for b in batch])

    # 1. RGB
    has_rgb = torch.tensor([b["rgb"] is not None for b in batch], dtype=torch.bool)
    out["has_rgb"] = has_rgb
    if has_rgb.any():
        rgb_shape = next(b["rgb"].shape for b in batch if b["rgb"] is not None)
        out["rgb"] = torch.stack([
            b["rgb"] if b["rgb"] is not None else torch.zeros(rgb_shape, dtype=torch.float32)
            for b in batch
        ])
    else:
        out["rgb"] = None

    # 2. Thermal
    has_thermal = torch.tensor([b["thermal"] is not None for b in batch], dtype=torch.bool)
    out["has_thermal"] = has_thermal
    if has_thermal.any():
        thermal_shape = next(b["thermal"].shape for b in batch if b["thermal"] is not None)
        out["thermal"] = torch.stack([
            b["thermal"] if b["thermal"] is not None else torch.zeros(thermal_shape, dtype=torch.float32)
            for b in batch
        ])
    else:
        out["thermal"] = None

    # 3. Electrical
    has_elec = torch.tensor([b["electrical_iv"] is not None for b in batch], dtype=torch.bool)
    out["has_electrical"] = has_elec
    if has_elec.any():
        out["electrical_iv"] = torch.stack([
            b["electrical_iv"] if b["electrical_iv"] is not None else torch.zeros((100, 2), dtype=torch.float32)
            for b in batch
        ])
        out["electrical_pv"] = torch.stack([
            b["electrical_pv"] if b["electrical_pv"] is not None else torch.zeros((100, 1), dtype=torch.float32)
            for b in batch
        ])
        out["electrical_scalars"] = torch.stack([
            b["electrical_scalars"] if b["electrical_scalars"] is not None else torch.zeros(7, dtype=torch.float32)
            for b in batch
        ])
        out["scalar_known_mask"] = torch.stack([
            b["scalar_known_mask"] if b["scalar_known_mask"] is not None else torch.zeros(7, dtype=torch.bool)
            for b in batch
        ])
    else:
        out["electrical_iv"] = None
        out["electrical_pv"] = None
        out["electrical_scalars"] = None
        out["scalar_known_mask"] = None

    # 4. Environment
    out["environment"] = torch.stack([
        b["environment"] if b["environment"] is not None else torch.zeros(3, dtype=torch.float32)
        for b in batch
    ])
    out["env_known_mask"] = torch.stack([
        b["env_known_mask"] if b["env_known_mask"] is not None else torch.zeros(3, dtype=torch.bool)
        for b in batch
    ])

    return out
