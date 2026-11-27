# Data — Canonical Schema and Adding Datasets

## Canonical Sample Schema

Every sample in the system, regardless of source, is represented as one record
conforming to this schema after adapter conversion (spec Section 7):

| Field | Type | Required | Notes |
|---|---|---|---|
| sample_id | str | YES | `{domain_id}__{original_id}`, globally unique |
| domain_id | str | YES | Domain-ID prefix (see datasets.md) |
| site_id | str | YES | Physical site; may equal domain_id for public datasets |
| panel_id | str\|null | conditional | Required for own-collected data; null OK for public |
| inspection_pass_id | str\|null | optional | Groups same-flight images |
| timestamp | str\|null | optional | ISO-8601 UTC |
| fault_label | list[str] | YES | Multi-label, subset of 7 canonical classes |
| rgb_path | str\|null | optional | Relative path under data/processed/ |
| thermal_path | str\|null | optional | Relative path under data/processed/ |
| thermal_raw_path | str\|null | optional | .npy radiometric array |
| electrical_path | str\|null | optional | .npz: iv_curve, pv_curve, scalars |
| environment | dict\|null | optional | {irradiance_w_m2, ambient_temp_c, wind_m_s} |
| split_hint | str\|null | optional | Official split from source dataset if provided |
| metadata | dict | YES | Must contain `license` key |

## Fault Taxonomy (FIXED — 7 classes)

1. `Healthy`
2. `Hotspot`
3. `Open/Short-circuit`
4. `Shading/Obstruction`
5. `Soiling`
6. `Structural (visual)`
7. `Electrical/Diode fault`

## Adding a New Dataset

1. Create adapter: `src/data/adapters/<dataset_id>.py` implementing `DatasetAdapter`.
2. Create label map: `src/data/label_maps/<dataset_id>.yaml`.
3. Add dataset_id to `data.adapters` list in `configs/base.yaml`.
4. Place raw data under `data/raw/<dataset_id>/`.
5. Run `python -m src.data.indexer --config configs/base.yaml`.

## Own-Collected Data (MVE-tier)

Place files under `data/raw/own_site_<n>/` with structure:
```
data/raw/own_site_a/
  metadata.yaml          # site_id, collection_date, camera_model, GPS
  rgb/<panel_id>/<timestamp>.jpg
  thermal/<panel_id>/<timestamp>.jpg
  electrical/<panel_id>/<timestamp>.npz  (optional)
  labels.csv             # columns: panel_id, timestamp, fault_label, inspection_pass_id
```
