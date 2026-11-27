# PROJECT_IMPLEMENTATION_SPEC.md
## Multimodal, Cross-Site, Synthetic-Data-Augmented PV Fault Detection — Implementation Specification

**Purpose of this document:** This is the single source of truth for building the complete repository. It is written to be executed by an AI coding agent (Google Antigravity) with minimal ambiguity, working from an empty project folder. Every architectural decision below is final unless explicitly marked `TO VERIFY`, `ASSUMPTION`, or `REQUIRES USER INPUT`.

**Source document:** `PV_Fault_Detection_Research_Blueprint.md` (research decisions preserved and operationalized below; not re-litigated except where noted).

---

## 1. Project Objective

Build a research codebase that trains and evaluates PV (photovoltaic) panel fault-classification models across three investigative axes, and produces the tables/figures needed to answer four fixed research questions with statistical rigor:

1. **Fusion** — does combining RGB + thermal + physics-derived electrical features beat the best single-modality model, in-domain and under domain shift?
2. **Cross-domain generalization** — does multimodality shrink the source→unseen-site generalization gap (not just raise absolute accuracy)?
3. **Synthetic data** — does physics-based + diffusion-based synthetic data narrow the generalization gap specifically, or only help in-distribution accuracy?
4. **Domain-generalization (DG) technique value** — do CORAL / adversarial-DG / domain-specific normalization beat a plain pooled-ERM baseline on leave-one-domain-out (LODO) evaluation?

The system must be able to run the full experiment ladder (Section 15) from public datasets alone (no real site data required to get a defensible result — see the trimmed scope in Section 6 of the source blueprint), and must be extensible to real multi-site data without any pipeline changes (Section 21, hardware interface).

---

## 2. Research Questions (Fixed — Do Not Modify)

- **RQ1 (Fusion value):** Does fusing RGB + thermal + electrical-signature features produce a statistically significant improvement in fault classification over the best single-modality baseline, both in-domain and under domain shift?
- **RQ2 (Generalization gap):** How large is the accuracy/F1 gap between source-domain and unseen-site performance for single- vs. multi-modality models, and does multimodality reduce this gap?
- **RQ3 (Synthetic data value):** Does physics-based + generative synthetic data reduce the cross-site generalization gap, under which regime (pretrain→finetune / blended ratio / domain randomization), and can negative transfer occur?
- **RQ4 (DG technique value):** Among CORAL / adversarial-DG / domain-specific normalization, which (if any) beats plain pooled-ERM on LODO evaluation?

These four questions are the acceptance test for the *research* layer (Section 24). Every experiment in the matrix (Section 15) exists to answer one of them.

---

## 3. Research Gap (Carried Forward, Do Not Re-derive)

No published PV fault-detection study combines (a) rigorous leave-one-site/dataset-out cross-domain evaluation with (b) genuine multimodal image+electrical fusion, and (c) a controlled ablation of whether synthetic data narrows the generalization gap specifically (as opposed to in-distribution accuracy only). DG methodology (CORAL/MMD/adversarial-DG/LODO benchmarking) is mature but imported from bearing/rotating-machinery diagnosis and essentially unapplied to PV image+electrical fusion. The codebase's job is to *operationalize* this gap into a reproducible benchmark, not to invent new algorithms.

---

## 4. Final Research Hypothesis (Fixed)

> H1: Feature-level multimodal fusion of RGB, thermal, and physics-derived electrical signatures, combined with domain-invariant representation learning trained across multiple heterogeneous source sites, will reduce the source-to-unseen-site generalization gap relative to the strongest single-modality baseline, and synthetic data will provide additional, but modality- and ratio-dependent, improvement rather than a uniform benefit.

This hypothesis must be falsifiable in the codebase's output: it must be possible for the results tables to show H1 is *wrong* (e.g., DG doesn't beat ERM, or synthetic data hurts generalization) without the pipeline needing to change. Negative/null results are valid outputs, not bugs.

---

## 5. Final Architecture — System-Level

```
RGB image (224x224x3)
    -> EfficientNet-B0 encoder (ImageNet-pretrained, separate weights from thermal)
    -> GAP -> projection head -> f_rgb (256-d)

Thermal image (224x224x1, + optional raw radiometric temperature channel 224x224x1)
    -> EfficientNet-B0 encoder (separate weights, ImageNet-init)
    -> GAP -> projection head -> f_thermal (256-d)

Electrical sequence:
    I-V curve resampled to 100 points (2 channels: I, V)
    + P-V derived curve (1 channel)
    + scalar features: Voc, Isc, Vmp, Imp, FF, Rs, Rsh (7-d, concatenated after encoder)
    + environmental conditioning: irradiance, ambient temp, wind (3-d, concatenated with scalars)
    -> 1D-CNN (3 conv blocks) -> 2-layer Transformer encoder over curve tokens
    -> f_elec (128-d, after concatenating scalar+env features and an MLP projection)

f_rgb, f_thermal -> Gated Cross-Attention block (bidirectional: RGB attends to thermal, thermal attends to RGB)
    -> fused image feature f_img (256-d)

f_img, f_elec -> concatenation -> MLP bottleneck -> shared representation z (256-d)
    -> [domain-invariance regularization: CORAL loss in-batch against other source domains;
        OPTIONAL gradient-reversal adversarial domain-classifier head, config-controlled]
    -> Fault classification head: Linear(256 -> 7), multi-label sigmoid (faults can co-occur)
    -> (training-only) Domain classifier head: Linear(256 -> num_domains), softmax, gradient-reversed
```

Environmental metadata is NOT a separate encoder branch — it is concatenated into the electrical branch input because it directly conditions I-V curve shape (this is a final decision from the source blueprint, Section 10).

**Fusion decision:** feature-level cross-attention for the two image branches; late (post-encoder) concatenation for the electrical branch, because electrical and image data have different acquisition cadence and must not be forced through a shared tokenizer (rejected alternative: single shared multimodal transformer — insufficient paired data at current dataset scale).

**Backbone alternative (config-selectable, not default):** ConvNeXt-Tiny may replace EfficientNet-B0 per-encoder via config; EfficientNet-B0 is the default or first implemented backbone because it is smaller and faster to iterate with.

**DG attachment point:** CORAL / adversarial-DG / domain-specific normalization all attach at the shared bottleneck `z` (256-d), never inside the modality-specific encoders. This keeps the DG module swappable without touching encoder or fusion code (see Section 12).

**Missing modality handling — final decision:** modality dropout during training (randomly zero out one image modality's contribution to the cross-attention block with configurable probability `modality_dropout_prob`) plus a learned "missing modality" embedding token substituted when a modality is structurally absent from a sample. This supports RGB-only, Thermal-only, RGB+Thermal, and full-trimodal samples through **one model**, rather than N separate models. The electrical branch degrades to a learned "no electrical data" embedding when absent (common for many public image-only datasets). Rationale: separate per-combination models would multiply the experiment matrix (Section 15) beyond budget, and modality dropout is standard practice for training under partial-modality data. `ASSUMPTION`: modality dropout probability default `0.15`; tune only if val metrics show it needed.

---

## 6. Dataset Strategy

### 6.1 Domain definition
A **domain** = a distinct (site x camera x climate) combination. Before any real multi-site data exists, the heterogeneous **public datasets themselves function as pseudo-domains** (different cameras/sites/geographies already). This is what makes Phase 1 (public-dataset LODO testbed) a self-sufficient starting point.

### 6.2 Datasets used (see Section 20 for acquisition detail: AUTOMATIC / MANUAL / OPTIONAL)
| Dataset | Modality | Role | Domain ID prefix |
|---|---|---|---|
| InfraredSolarModules (Raptor Maps) | Thermal | Core pseudo-domain | `irsm` |
| Alfaro-Mejia Cali thermal set | Thermal | Core pseudo-domain (genuinely distinct climate/camera) | `cali` |
| SolarFCD | RGB+Thermal (unified) | Core pseudo-domain, multi-source | `solarfcd` |
| RGBIRPV | RGB+IR paired | Core pseudo-domain (paired) | `rgbirpv` |
| Bimodal UAV dataset (YOLOv11-MO) | RGB+Thermal | Core pseudo-domain | `bimodal_uav` |
| PVMD (via SolarFCD) | Thermal | Folded into SolarFCD | (see solarfcd) |
| MATLAB/Simulink I-V simulation methodology | Electrical (synthetic, regenerated) | Electrical branch primary data source (regenerated via `pvlib`, not downloaded) | `synth_elec` |
| LTSpice 2-diode GaAs/Ge dataset | Electrical (synthetic) | Reference/validation only, not primary (III-V cells, not representative Si) | `ltspice_ref` |
| NREL PVDAQ / Sandia PV fleet data | Electrical+environmental, real, healthy-dominant | Healthy-domain diversity + irradiance/temp normalization + realistic conditioning for fault injection | `nrel_pvdaq` |
| ELPV / E-ELPV | EL | Optional lab sub-study ONLY — never part of the core field-system taxonomy | `elpv` (excluded from core `configs/*.yaml` by default) |

`TO VERIFY`: exact current license/availability of every academic-paper dataset (RGBIRPV, Bimodal UAV, SolarFCD, LTSpice set) must be re-checked at the source before pipeline work begins — several are "on request" or supplementary-material-only, not a standing public URL. The codebase must not assume any URL below is still valid at build time; the dataset acquisition script must fail loudly (not silently) if a listed source is unreachable, and record the failure in `data/DATASET_STATUS.md` rather than fabricating data.

### 6.3 Real-world data collection (parallel track, not blocking)
Own multi-site collection follows the MVE tier from the source blueprint (2 sites, RGB + thermal, synthetic-only electrical validated by spot-check multimeter readings). The repository must accept this data via the same `data/raw/<dataset_name>/` + adapter interface as public datasets (Section 9) — no special-cased code path.

---

## 7. Canonical Data Schema

Every sample in the system, regardless of source dataset, is represented as one record conforming to this schema after adapter conversion. This is the *only* format the model layer ever sees.

```
sample_id:        str   REQUIRED  — globally unique, format "{domain_id}__{original_id}"
domain_id:        str   REQUIRED  — one of the domain-ID prefixes in Section 6.2; the unit LODO splits on
site_id:          str   REQUIRED  — physical site identifier; may equal domain_id for single-site public datasets
panel_id:         str | null      — physical panel identifier; REQUIRED for real-collected data, null allowed
                                     for public datasets that don't expose panel identity (leakage checks then
                                     fall back to inspection-pass-level grouping, see Section 9.5)
inspection_pass_id: str | null    — groups images captured in the same pass/flight, for leakage prevention
timestamp:         str | null     — ISO-8601 UTC; null allowed if source dataset has no timestamp
fault_label:       list[str]      REQUIRED — multi-label, subset of the 7-class taxonomy (Section 7.1);
                                     minimum length 1 (a "Healthy" sample has fault_label = ["Healthy"])
rgb_path:          str | null     — relative path under data/processed/; null if modality absent
thermal_path:      str | null     — relative path under data/processed/; null if modality absent
thermal_raw_path:  str | null     — relative path to raw radiometric temperature array (.npy), if available
electrical_path:   str | null     — relative path to a serialized electrical record (.npz: iv_curve, pv_curve,
                                     scalar features), null if modality absent
environment:       dict | null    — {irradiance_w_m2: float|null, ambient_temp_c: float|null, wind_m_s: float|null}
split_hint:        str | null     — "train"|"val"|"test" ONLY if the source dataset ships an official split;
                                     the canonical splitter (Section 9.5) does not have to honor this, but must
                                     record when it overrides it
metadata:          dict           — free-form, source-specific passthrough (original filenames, camera model,
                                     original label strings pre-mapping, dataset license tag)
```

**Types/units:** `irradiance_w_m2` in W/m^2, `ambient_temp_c` in Celsius, `wind_m_s` in m/s, timestamps always UTC. Electrical scalar features (Voc, Isc, Vmp, Imp, FF, Rs, Rsh) are stored in the `.npz` electrical record, not inline in the JSON/parquet index, to keep the index small.

**Missing-data behavior:** any `*_path` field being `null` means that modality is structurally absent for that sample — the dataloader must substitute the learned missing-modality embedding (Section 5), never impute pixels/curves. `environment` fields may be individually null (e.g., wind unknown) — the electrical encoder's environment-conditioning sub-vector uses a per-field learned "unknown" embedding for null environment values, not zero-imputation (zero has physical meaning here — zero irradiance is a valid value).

**Storage format:** the canonical index (one row per sample, all fields above except paths-to-large-arrays) is a single Parquet file per dataset stage: `data/processed/index/{split_stage}.parquet` where `split_stage` in `{full, train, val, test, lodo_<domain_id>}`. Large arrays (images, electrical curves) are never inlined in the index.

### 7.1 Fault Taxonomy (Fixed, 6 classes + Healthy = 7 total)
`Healthy`, `Hotspot`, `Open/Short-circuit`, `Shading/Obstruction`, `Soiling`, `Structural (visual)`, `Electrical/Diode fault`. Cell-level crack/PID/snail-trail fine-grained detail is deferred entirely to the optional EL sub-study (excluded from the core taxonomy — see Section 6.2, ELPV row). The label-mapping table from every source dataset's native taxonomy to these 7 classes lives in `src/data/label_maps/{dataset_id}.yaml` (one file per adapter) and must be reviewed by a human before first use — mark unmapped source classes as `TO VERIFY` in that file rather than guessing.

---

## 8. Repository Tree (Final)

```text
pv_fault_detection/
|
├── README.md                       # project overview, quickstart, links to docs/
├── pyproject.toml                  # single source of dependency truth (Section 19)
├── .gitignore                      # data/, checkpoints/, logs/, results/*, .env, __pycache__
├── .env.example                    # placeholders for any credentialed dataset access (NREL API key, etc.)
├── STATUS.md                       # living build-status file (Antigravity updates this every stage, Section 26)
|
├── configs/                        # ALL experiment configs; see Section 18 for schema
│   ├── base.yaml                   # shared defaults every experiment config inherits from
│   ├── baseline_rgb.yaml
│   ├── baseline_thermal.yaml
│   ├── baseline_electrical.yaml
│   ├── multimodal_rgb_thermal.yaml
│   ├── multimodal_full.yaml
│   ├── fusion_late_vs_crossattn.yaml
│   ├── synthetic_ablation.yaml     # sweeps synthetic ratio via CLI override, not 5 separate files
│   ├── domain_generalization.yaml  # sweeps DG technique via CLI override
│   └── lodo_final.yaml             # final Level-6 system, Section 15 row 9
│
├── data/
│   ├── raw/                        # untouched downloaded/collected data, one subfolder per dataset_id
│   │   └── <dataset_id>/           # e.g. data/raw/irsm/, data/raw/cali/, data/raw/own_site_a/
│   ├── processed/                  # canonical-format output of adapters (Section 9)
│   │   ├── images/<dataset_id>/
│   │   ├── electrical/<dataset_id>/
│   │   └── index/                  # canonical Parquet index files (Section 7)
│   ├── synthetic/                  # generator output, kept physically separate from processed/
│   │   ├── electrical/
│   │   └── images/
│   └── DATASET_STATUS.md           # generated by acquisition script; which datasets are present/missing/failed
│
├── notebooks/                      # exploratory only; NEVER imported by src/; one-off EDA, sanity checks
│   └── 00_eda_template.ipynb
│
├── src/
│   ├── __init__.py
│   ├── common/                     # shared utilities with no ML-specific logic
│   │   ├── config.py               # config loading/merging/validation (Section 18)
│   │   ├── seeding.py              # deterministic seeding (Section 16)
│   │   ├── logging_setup.py        # structured logging + experiment tracker init (Section 20)
│   │   └── paths.py                # canonical path resolution, never hard-code paths elsewhere
│   │
│   ├── data/                       # dataset handling: download, validate, index, split
│   │   ├── adapters/
│   │   │   ├── base.py             # DatasetAdapter abstract interface (Section 9)
│   │   │   ├── irsm.py
│   │   │   ├── cali.py
│   │   │   ├── solarfcd.py
│   │   │   ├── rgbirpv.py
│   │   │   ├── bimodal_uav.py
│   │   │   ├── nrel_pvdaq.py
│   │   │   ├── elpv.py             # optional sub-study only; not imported by core configs
│   │   │   └── own_site.py         # generic adapter for MVE-tier real collected data
│   │   ├── download.py             # AUTOMATIC-only downloads (Section 20); MANUAL ones print instructions
│   │   ├── validate.py             # schema validation against Section 7 canonical schema
│   │   ├── indexer.py              # builds data/processed/index/full.parquet from all adapters
│   │   └── label_maps/*.yaml       # per-dataset native-label -> canonical-taxonomy mapping (Section 7.1)
│   │
│   ├── preprocessing/
│   │   ├── rgb.py                  # resize/normalize/blur-filter (Laplacian variance)
│   │   ├── thermal.py              # emissivity/reflected-temp correction, radiometric handling
│   │   ├── electrical.py           # resampling to 100 pts, STC normalization, scalar-feature extraction
│   │   ├── environment.py          # irradiance/temp/wind cleaning + null-handling
│   │   ├── synchronization.py      # UTC timestamp alignment, +/-2 min tolerance pairing
│   │   └── quality_control.py      # automated irradiance/wind validity filtering (IEC 62446-3-style)
│   │
│   ├── splitting/
│   │   ├── random_split.py         # debugging only, explicitly labeled unsafe for reporting
│   │   ├── grouped_split.py        # group by panel_id / inspection_pass_id
│   │   ├── site_held_out_split.py  # main cross-domain single-held-out-site split
│   │   └── lodo_split.py           # leave-one-domain-out rotation generator (Section 9.5)
│   │
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── image_encoder.py    # shared EfficientNet-B0 / ConvNeXt-Tiny wrapper (timm-based)
│   │   │   └── electrical_encoder.py # 1D-CNN + Transformer encoder
│   │   ├── fusion/
│   │   │   ├── cross_attention.py  # gated bidirectional cross-attention (image branches)
│   │   │   └── late_fusion.py      # concatenation fusion for electrical branch + late-fusion ablation baseline
│   │   ├── domain_generalization/
│   │   │   ├── coral.py
│   │   │   ├── adversarial_dg.py   # gradient-reversal domain classifier
│   │   │   └── domain_norm.py      # domain-specific batch/instance normalization
│   │   ├── heads/
│   │   │   ├── classifier_head.py  # multi-label sigmoid fault head
│   │   │   └── domain_head.py      # training-only domain classifier
│   │   └── pv_fault_model.py       # top-level nn.Module assembling all of the above per config
│   │
│   ├── synthetic/
│   │   ├── electrical_sim.py       # pvlib single/two-diode fault simulator (Section 11)
│   │   ├── image_diffusion.py      # DDPM/Stable-Diffusion-inpainting defect augmentation, SSIM-filtered
│   │   ├── image_gan_baseline.py   # GAN comparison baseline only, not primary
│   │   ├── quality_control.py      # SSIM filtering, Voc/Isc/FF physical-range validation
│   │   └── mixing.py               # real/synthetic ratio mixing, train-split-only conditioning enforcement
│   │
│   ├── training/
│   │   ├── trainer.py              # single training loop, config-driven, used by every experiment type
│   │   ├── losses.py               # multi-label focal/class-weighted loss, CORAL loss, adversarial loss
│   │   └── optim.py                # optimizer/scheduler construction from config
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # macro-F1, per-class recall, ECE, generalization gap, etc. (Section 13)
│   │   ├── lodo_runner.py          # runs the full LODO rotation, aggregates per Section 9.5 protocol
│   │   ├── statistical_tests.py    # bootstrap CIs, Wilcoxon signed-rank, effect size
│   │   └── ablation_runner.py      # orchestrates the 9-row ablation matrix (Section 15)
│   │
│   ├── visualization/
│   │   ├── confusion_matrix.py
│   │   ├── roc_pr_curves.py
│   │   ├── domain_gap_plots.py
│   │   ├── feature_viz.py          # embedding-space visualization (e.g. UMAP of shared bottleneck z)
│   │   ├── thermal_viz.py
│   │   ├── iv_curve_viz.py
│   │   └── ablation_plots.py
│   │
│   └── experiments/
│       ├── run.py                  # single entrypoint: python -m src.experiments.run --config ...
│       └── registry.py             # experiment-id <-> config <-> result-dir bookkeeping (Section 16)
│
├── scripts/                        # thin CLI wrappers around src/ modules; no logic lives here
│   ├── prepare_data.sh
│   ├── run_experiment.sh
│   ├── run_ablation_matrix.sh
│   ├── run_lodo.sh
│   └── generate_report_tables.sh
│
├── experiments/                    # git-ignored EXCEPT experiment registry; actual run configs snapshotted here
│   └── registry.json               # append-only log of every run: id, config hash, git commit, timestamp
│
├── tests/
│   ├── unit/
│   │   ├── test_adapters.py
│   │   ├── test_preprocessing.py
│   │   ├── test_electrical_repr.py
│   │   ├── test_label_mapping.py
│   │   ├── test_model_forward.py
│   │   ├── test_fusion.py
│   │   ├── test_synthetic_generator.py
│   │   └── test_splitting.py
│   ├── integration/
│   │   ├── test_raw_to_dataloader.py
│   │   ├── test_dataloader_to_model.py
│   │   ├── test_train_one_step.py
│   │   ├── test_evaluate.py
│   │   └── test_checkpoint_roundtrip.py
│   └── leakage/
│       ├── test_no_panel_crosses_split.py
│       ├── test_no_site_in_train_and_lodo_test.py
│       └── test_synthetic_not_from_test_split.py
│
├── checkpoints/                    # git-ignored; {experiment_id}/checkpoint_best.pt, checkpoint_last.pt
├── logs/                           # git-ignored; structured JSON logs + tracker-specific files
├── results/                        # NOT git-ignored for final headline results; per-experiment subfolder (Section 17)
├── reports/                        # generated tables (Section 22) and the paper-structure draft (optional)
└── docs/
    ├── architecture.md
    ├── data.md
    ├── datasets.md
    ├── training.md
    ├── evaluation.md
    ├── synthetic_data.md
    └── domain_generalization.md
```

**What must NOT belong where:**
- `notebooks/` — never imported by `src/`; exploratory only; no notebook is a dependency of any pipeline stage.
- `scripts/` — no business logic; if a script grows past ~30 lines of real logic, that logic belongs in `src/` and the script becomes a thin `argparse` + import wrapper.
- `data/raw/` — never written to by any code after initial download/copy; treated as read-only once populated.
- `checkpoints/`, `logs/` — never committed to git; `results/` for headline/reportable results IS committed (small: metrics.json, tables, plots — not raw predictions or checkpoints).
- `configs/` — no Python logic, YAML only.


## 9. Module Specifications

### 9.1 `src/data/adapters/base.py` — `DatasetAdapter` interface
- **Purpose:** every dataset (public or own-collected) implements this interface; the rest of the system never has dataset-specific branches outside `src/data/adapters/`.
- **Interface (abstract methods every adapter implements):**
  - `dataset_id() -> str` — e.g. `"irsm"`.
  - `validate_raw() -> bool` — checks `data/raw/<dataset_id>/` has the expected structure; returns False (not raises) on mismatch so callers can report status, not crash.
  - `iter_samples() -> Iterator[RawSample]` — yields one record per source-native sample, before canonical mapping.
  - `to_canonical(raw_sample) -> CanonicalSample` — applies `label_maps/<dataset_id>.yaml`, fills the Section 7 schema, sets `domain_id` to this adapter's fixed prefix.
  - `license_tag() -> str` — recorded into `metadata.license` for every sample; used by `data/DATASET_STATUS.md`.
- **Inputs:** raw files under `data/raw/<dataset_id>/`.
- **Outputs:** canonical-schema records appended to `data/processed/index/full.parquet`, plus copied/converted media under `data/processed/images|electrical/<dataset_id>/`.
- **Tested by:** `tests/unit/test_adapters.py` (one test class per adapter, using a small fixture subset, never the full dataset).
- **Called by:** `src/data/indexer.py` only.

### 9.2 `src/data/indexer.py`
- **Purpose:** run every registered adapter, concatenate canonical output, write the full index.
- **Inputs:** none (discovers adapters via a registry list; `EL PV` adapter excluded from the default registry unless `--include-el-substudy` is passed).
- **Outputs:** `data/processed/index/full.parquet`, `data/DATASET_STATUS.md` (which datasets present/absent/failed validation).
- **Config:** `configs/base.yaml` -> `data.adapters: [irsm, cali, solarfcd, rgbirpv, bimodal_uav, nrel_pvdaq]` (list is config-controlled, not hard-coded, so the trimmed-scope project from the source blueprint's Section 27 is a config change, not a code change).
- **CLI:** `python -m src.data.indexer --config configs/base.yaml`.

### 9.3 `src/preprocessing/*`
Each module (`rgb.py`, `thermal.py`, `electrical.py`, `environment.py`) is a pure-function library: `def process(canonical_sample_field, params) -> processed_field`. No module reads YAML directly — parameters are passed in from the caller (trainer/dataloader construction), which reads config once. This keeps preprocessing unit-testable without a config file (see `tests/unit/test_preprocessing.py`).
- `synchronization.py`: input = list of candidate (image_timestamp, electrical_timestamp) pairs per site; output = matched pairs within +/-2 min, unmatched images get `electrical_path=null`.
- `quality_control.py`: input = raw thermal image + irradiance/wind reading; output = boolean validity flag per IEC 62446-3-style thresholds (irradiance >= 600 W/m^2, wind < 5 m/s — config-overridable, `ASSUMPTION` defaults from source blueprint Section 5.2).

### 9.4 `src/models/pv_fault_model.py`
- **Purpose:** single top-level `nn.Module` that assembles encoders + fusion + DG module + heads according to config — this is the ONLY place architecture is assembled; encoders/fusion/DG modules are never instantiated ad hoc elsewhere.
- **Inputs (forward pass):** a batch dict with keys `rgb`, `thermal`, `thermal_raw`, `electrical_iv`, `electrical_pv`, `electrical_scalars`, `environment`, each possibly containing a missing-modality mask.
- **Outputs:** `{"fault_logits": Tensor[B,7], "domain_logits": Tensor[B,num_domains] | None, "z": Tensor[B,256]}` — `z` is exposed for CORAL loss computation and embedding visualization.
- **Config keys consumed:** `model.image_backbone`, `model.fusion_type` (`cross_attention`|`late`), `model.dg_method` (`none`|`coral`|`adversarial`|`domain_norm`), `model.modality_dropout_prob`, `model.modalities` (subset list, e.g. `[rgb, thermal]` for single/dual-modality ablation rows).
- **Tested by:** `tests/unit/test_model_forward.py` (forward pass shape checks across every `modalities` subset), `tests/unit/test_fusion.py` (cross-attention vs late fusion output shape/gradient-flow checks).

### 9.5 `src/splitting/lodo_split.py`
- **Purpose:** implement the exact LODO protocol from the source blueprint Section 9.
- **Inputs:** canonical index Parquet, `domain_id` column.
- **Outputs:** for each domain `Di`, writes `data/processed/index/lodo_<Di>_train.parquet` (all other domains) and `lodo_<Di>_test.parquet` (only `Di`, held out entirely — zero exposure during training/val/hyperparameter tuning).
- **Fallback rule:** if `< 3` genuinely distinct domains exist in the active config, raise a clear error directing the caller to use `site_held_out_split.py` with domain-stratified k-fold instead — LODO with fewer than 3 domains is not implemented as a silent fallback; it must be an explicit config choice.
- **Grouping key:** splits by `panel_id` where present, else by `inspection_pass_id`, else by `sample_id` with a loud warning that leakage-prevention is degraded for that dataset (this only happens for public datasets with no panel/pass identifiers, e.g. `irsm` crops).


## 10. Model Specifications (Precise, Per Component)

| Component | Backbone/method | Pretrained | Input size | Output dim | Notes |
|---|---|---|---|---|---|
| RGB encoder | EfficientNet-B0 (`timm`) | ImageNet | 224x224x3 | 256-d (after GAP + projection head) | separate weights from thermal encoder |
| Thermal encoder | EfficientNet-B0 (`timm`), separate instance | ImageNet init | 224x224x1(+1 raw temp channel optional) | 256-d | first conv layer adapted for 1 or 2 input channels |
| Electrical encoder | 1D-CNN (3 conv blocks, channels 2->32->64->128) + 2-layer Transformer encoder (d_model=128, nhead=4) over 100 curve tokens | none (trained from scratch — electrical data is synthetic/limited, no pretrained electrical backbones exist) | I-V (100,2) + P-V (100,1) | 128-d sequence pooling, then concatenated with 7 scalar + 3 env features -> MLP -> 128-d `f_elec` | scalar/env features projected via small MLP before concatenation |
| Cross-attention fusion | Gated bidirectional cross-attention, 2 heads, hidden 256 | n/a | f_rgb (256), f_thermal (256) | f_img (256) | gate = learned sigmoid scalar per sample controlling attention contribution; degrades gracefully to identity when one modality is the missing-modality embedding |
| Late fusion (electrical) | Concatenation + MLP bottleneck | n/a | f_img (256) + f_elec (128) | z (256) | MLP: Linear(384->256) -> LayerNorm -> GELU -> Dropout(0.2) |
| DG module | CORAL / adversarial (gradient-reversal) / domain-norm — config-selected, exactly one active at a time (or `none`) | n/a | z (256) | z (256, unchanged shape; loss-only for CORAL/adversarial, in-place normalization change for domain-norm) | see Section 12 |
| Classifier head | Linear(256->7) | n/a | z (256) | 7 sigmoid logits (multi-label) | class-weighted / focal loss, Section 13.2 |
| Domain head (train-only) | Linear(256->num_domains), gradient-reversal layer upstream | n/a | z (256) | num_domains softmax logits | discarded at inference; only present if `model.dg_method == adversarial` |

**Normalization/activation/dropout defaults (`ASSUMPTION`, config-overridable):** GELU activations in all fusion/bottleneck MLPs, LayerNorm (not BatchNorm) in the bottleneck (BatchNorm reserved for `domain_norm` DG method specifically, where per-domain BatchNorm statistics are the mechanism itself), Dropout 0.2 in bottleneck and classifier head, Dropout 0.1 in the electrical Transformer.

**Loss function:** multi-label focal loss (config: `alpha`, `gamma`, default `gamma=2.0`) OR class-weighted BCE (config-selectable, `training.loss_type`), + optional CORAL loss term (weight `training.coral_weight`) + optional adversarial domain-classifier loss term (weight `training.adversarial_weight`, gradient-reversal lambda scheduled per standard DANN warm-up schedule). Total loss = `fault_loss + coral_weight * coral_loss + adversarial_weight * domain_loss` (terms zeroed out when their DG method is inactive).

---

## 11. Synthetic-Data System

**Module home:** `src/synthetic/`. Two independent generators, one quality-control gate, one mixing module — never entangled.

### 11.1 Electrical synthetic generator — `electrical_sim.py`
- **What is generated:** I-V/P-V curve pairs + derived scalar features (Voc, Isc, Vmp, Imp, FF, Rs, Rsh) for all 7 fault classes plus Healthy.
- **Why:** electrical branch's primary data source (real electrical ground truth is scarce; physics simulation is decades-validated).
- **Method:** `pvlib-python` single/two-diode equivalent-circuit model — use `pvlib`, never hand-rolled diode equations (explicit decision from source blueprint Section 13).
- **Parameters:** 5 single-diode parameters (`I_L`, `I_0`, `R_s`, `R_sh`, `n`) perturbed per fault class per a documented fault-injection table (`src/synthetic/fault_injection_table.yaml` — e.g. Hotspot: elevated `R_s` in one sub-cell + reduced `I_L`; Open-circuit: `R_s -> inf` proxy; PID: `R_sh` reduction over simulated time). This table is the actual mathematical/algorithmic spec and must be filled from the literature values cited in the source blueprint Section 2.4/2.11 (35,000-curve MATLAB/Simulink methodology) — `TO VERIFY` exact per-fault parameter ranges against that paper before first run; ship conservative literature-typical defaults and flag them `ASSUMPTION` in code comments.
- **Realism control:** every generated curve is validated against known Voc/Isc/FF physical ranges (Section 11.3) before acceptance; irradiance/temperature conditioning drawn from real NREL PVDAQ healthy-domain distributions (Section 6.2) rather than uniform random, so synthetic curves reflect realistic operating conditions.
- **Storage:** `data/synthetic/electrical/<fault_class>/<uuid>.npz` (iv_curve, pv_curve, scalars, generation_params, source_conditioning_domain_id).
- **Labelling:** ground-truth fault label is set by construction (it's the parameter perturbation applied) — never inferred.

### 11.2 Image synthetic generator — `image_diffusion.py` (primary), `image_gan_baseline.py` (comparison only)
- **What:** DDPM/Stable-Diffusion-style inpainting defect augmentation on RGB/thermal, targeting minority fault classes.
- **Why:** class-imbalance correction for structural/electrical-fault classes with few real examples (source blueprint Section 2.16 table: <100 examples for some classes in SolarFCD).
- **Method:** `diffusers` (Hugging Face) DDPM inpainting conditioned on a real training-split image + a synthetic defect mask; NOT full image generation from noise (preserves background structure — spatial-identity-loss style, per source blueprint Section 2.11).
- **Realism control:** SSIM filtering against the source real image — reject generated samples below `synthetic.ssim_min_threshold` (config, `ASSUMPTION` default 0.75).
- **GAN baseline:** `image_gan_baseline.py` implements a comparable DCGAN/SIGAN-style generator for the Section 15 comparison row only; never used as the primary augmentation source (source blueprint explicitly ranks diffusion over GAN for training stability).
- **Storage:** `data/synthetic/images/<modality>/<fault_class>/<uuid>.png` + a sidecar JSON recording `source_real_sample_id` (the training-split real image it was conditioned on) — this sidecar is what the leakage test (`tests/leakage/test_synthetic_not_from_test_split.py`) checks against.

### 11.3 Quality control — `quality_control.py`
- Electrical: reject curves with Voc/Isc/FF outside a documented physical range table (`src/synthetic/valid_ranges.yaml`, `TO VERIFY` against literature-typical Si-module ranges).
- Images: SSIM filter (above) + basic sanity (non-blank, correct channel count).
- Both paths log rejection rate to the experiment tracker (Section 20) — a silently-high rejection rate is itself a diagnostic signal worth surfacing, not hiding.

### 11.4 Mixing — `mixing.py`
- **Contamination prevention (hard rule, enforced in code, not just documentation):** synthetic samples are generated ONLY by conditioning on training-split real samples; the function signature `generate(...)` requires a `split` argument and raises `ValueError` if called with `split != "train"`. This directly implements the leakage rule from the source blueprint Section 14.
- **Mixing ratios supported (config `synthetic.ratio`):** `0.0` (real-only), `0.1, 0.25, 0.5, 1.0, "unlimited"` (synthetic-only), plus a `pretrain_finetune` mode (two-stage: train on synthetic-only, then fine-tune on real-only) and `domain_randomized` mode (synthetic electrical conditioning irradiance/temperature/camera-noise parameters swept beyond the observed real range).
- **Evaluation coupling:** `mixing.py` never computes metrics itself — it only produces a mixed dataloader; whether a ratio "helps" is determined downstream by `evaluation/metrics.py`'s generalization-gap computation (Section 13), per the source blueprint's explicit warning that in-domain accuracy improving is not sufficient evidence of a real benefit.

---

## 12. Domain-Generalization (DG) System

**Module home:** `src/models/domain_generalization/`. Exactly one method active per experiment (config `model.dg_method`), always compared against the mandatory ERM baseline (`dg_method: none`) — ERM must be beaten to justify any method's added complexity (RQ4).

| Method | File | Mechanism | Where attached | Loss/hyperparameters |
|---|---|---|---|---|
| ERM (baseline) | n/a | plain pooled training across all source domains | n/a | no additional loss term |
| CORAL | `coral.py` | second-order feature covariance alignment across source domains, computed in-batch | shared bottleneck `z` | `training.coral_weight` (float, `ASSUMPTION` default 1.0, tune via Optuna) |
| Adversarial DG | `adversarial_dg.py` | gradient-reversal layer + domain classifier head | shared bottleneck `z` | `training.adversarial_weight`, gradient-reversal lambda schedule (`training.grl_lambda_schedule`: linear warm-up over first 30% of training, standard DANN recipe) |
| Domain-specific normalization | `domain_norm.py` | per-domain BatchNorm statistics in the bottleneck MLP | shared bottleneck `z` | no extra loss term; purely architectural — swaps LayerNorm for domain-indexed BatchNorm |
| MMD (comparison only) | `coral.py` (shared file, alternate loss fn) | maximum mean discrepancy alignment | shared bottleneck `z` | `training.mmd_weight` |
| Test-time adaptation (comparison only) | separate eval-time-only hook in `evaluation/lodo_runner.py`, not a training-time module | BatchNorm statistic recalibration using a small unlabeled sample from the unseen domain at eval time | inference only | no training change; documented as a distinct evaluation mode, not a `model.dg_method` value |

**Training procedure:** DG methods are trained identically to ERM except for the added loss term(s) — this is enforced by construction: `src/training/trainer.py` has one training loop; DG method only changes what `pv_fault_model.py` returns and what `losses.py` adds, never the loop structure itself. This guarantees fusion/DG/synthetic-data axes stay orthogonal and combinable via config, not via code branches.

**Evaluation:** DG methods are evaluated exclusively via the LODO protocol (`lodo_runner.py`) — DG has no meaning under a random/in-domain split, so `configs/domain_generalization.yaml` hard-requires `evaluation.protocol: lodo` and the config loader (Section 18) rejects the combination `dg_method != none` with `protocol: random_split` at load time (fail fast, not silently ignore).


## 13. Training System

**Module home:** `src/training/`. One trainer (`trainer.py`), config-driven for every experiment type in the matrix — there is no "baseline trainer" vs "multimodal trainer" vs "DG trainer"; they are the same code path with different config.

### 13.1 Pipeline (fixed order, matches source blueprint Section 13)
```
Raw data -> Synchronization -> Cleaning -> Calibration -> Quality control -> Labeling
   -> Preprocessing -> Augmentation -> Synthetic generation/mixing
   -> Domain-aware split (site-level, never sample-level; LODO rotation)
   -> Modality-specific preprocessing -> Model training (staged, per Research Ladder Section 15.1)
   -> Hyperparameter optimization (Optuna, budget-limited) -> Model selection (best mean LODO macro-F1)
   -> Evaluation
```
Stages 1-8 (through domain-aware split) are handled by `src/data/` + `src/preprocessing/` + `src/splitting/` + `src/synthetic/`, run once and cached to `data/processed/` and `data/synthetic/` — the trainer never re-runs them; it consumes the cached, already-split Parquet indices.

### 13.2 Losses — `losses.py`
- `multilabel_focal_loss(logits, targets, alpha, gamma)` — primary, default.
- `class_weighted_bce_loss(logits, targets, class_weights)` — alternative, config-selectable.
- `coral_loss(z_source_batches)`, `mmd_loss(...)`, `adversarial_domain_loss(domain_logits, domain_targets)` — added conditionally per `model.dg_method` (Section 12).
- Total loss assembly happens in `trainer.py`, not inside the model — the model returns raw logits/features only, keeping loss policy swappable without touching `pv_fault_model.py`.

### 13.3 Optimizer/scheduler — `optim.py`
- AdamW default (`ASSUMPTION`, config-overridable), cosine LR schedule with linear warmup, `training.lr`, `training.weight_decay`, `training.warmup_steps` all config keys.

### 13.4 Hyperparameter search
- Optuna, budget-limited (config `optuna.n_trials`, `ASSUMPTION` default 30 per experiment family), objective = mean LODO macro-F1 (never single-split accuracy — this is a fixed decision from the source blueprint Section 13). Search space defined per-config in a `optuna_search_space` block; only run for the Level-6 final configuration and DG hyperparameters (coral_weight, adversarial_weight) — not for every ablation row, to keep compute bounded.

### 13.5 Staged training (Research Ladder — fixed correspondence to experiment matrix, Section 15)
Levels 1-6 from the source blueprint map directly to ablation-matrix rows; the trainer does not implement "levels" as a separate concept — each level is just a specific config (Section 18).

---

## 14. Evaluation System

**Module home:** `src/evaluation/`.

### 14.1 Metrics — `metrics.py`
Per-class and overall: accuracy, precision, recall, F1, **macro-F1 (primary, given class imbalance)**, balanced accuracy, ROC-AUC, PR-AUC, full confusion matrix, per-class recall (flagged especially for Hotspot/Open-Short — safety-critical), false-positive rate, expected calibration error (ECE), inference latency (ms/sample on the eval hardware), model size (param count + MB).

**Cross-domain-specific (computed only when `evaluation.protocol: lodo`):**
- `generalization_gap = mean(source_held_out_macro_F1) - unseen_site_macro_F1`, per LODO rotation and averaged.
- `relative_degradation = generalization_gap / source_performance`.
- `worst_case_domain_performance = min(macro_F1 across all LODO rotations)`.

### 14.2 LODO runner — `lodo_runner.py`
- **Inputs:** a trained-model factory (fresh model per rotation — DG methods must not see the held-out domain during any rotation's training), the LODO split indices from `src/splitting/lodo_split.py`.
- **Procedure:** for each domain `Di`, train fresh on all-other-domains train split (with internal val split carved from the training domains only, never from `Di`), evaluate once on `Di`'s test set, record all Section 14.1 metrics, repeat for every domain, aggregate mean +/- std.
- **Output:** `results/<experiment_id>/lodo_results.json` (per-rotation) + `results/<experiment_id>/lodo_summary.json` (aggregated).
- **Test-time adaptation mode:** optional flag `--tta` recalibrates BatchNorm statistics using a small unlabeled sample from `Di` before evaluation (Section 12) — reported as a separate row, never silently blended into the main LODO number.

### 14.3 Statistical validation — `statistical_tests.py`
- Minimum 5 seeds per configuration (config `training.n_seeds`, `ASSUMPTION` default 5, may drop to 3 under compute constraints per source blueprint Section 17 — this is a config value, not a hard-coded constant).
- Bootstrap resampling (1,000 resamples) for 95% CI on every headline number.
- Wilcoxon signed-rank test (paired, across seeds/domains) for any two-configuration comparison (e.g. fusion vs. single-modality) — never a bare point-estimate diff claim in generated tables.
- Effect size (rank-biserial correlation) reported alongside every p-value.

### 14.4 Ablation runner — `ablation_runner.py`
Orchestrates the 9-row matrix (Section 15) end to end: reads `configs/*.yaml`, applies the correct override for each row, calls `trainer.py` or `lodo_runner.py` as appropriate, writes one `results/<experiment_id>/` directory per row x seed x (domain rotation, if LODO).

---

## 15. Experiment Matrix (Final — 9 Rows, Fixed)

| # | Row | Protocol | Modalities | DG method | Synthetic ratio | Answers |
|---|---|---|---|---|---|---|
| 1 | RGB only | random split (in-domain) | `[rgb]` | none | 0.0 | Research-ladder Level 1/2 floor |
| 2 | Thermal only | random split | `[thermal]` | none | 0.0 | Research-ladder Level 1/2 floor |
| 3 | Electrical only | random split | `[electrical]` | none | 0.0 | Research-ladder Level 1/2 floor |
| 4 | RGB + Thermal | random split | `[rgb, thermal]` | none | 0.0 | Level 3 — isolates fusion contribution |
| 5 | RGB + Thermal + Electrical (full model) | random split | `[rgb, thermal, electrical]` | none | 0.0 | RQ1 in-domain half |
| 6 | Late fusion vs cross-attention | random split | `[rgb, thermal, electrical]` | none | 0.0 | fusion-mechanism comparison, fixed modality set |
| 7 | Real-only vs Real+Synthetic (ratio sweep) | random split, best arch from #5/#6 | `[rgb, thermal, electrical]` | none | sweep: 0, 0.1, 0.25, 0.5, 1.0, unlimited, pretrain_finetune, domain_randomized | RQ3 in-domain half; Level 4 |
| 8 | ERM vs CORAL vs Adversarial-DG vs domain-norm | **LODO** | `[rgb, thermal, electrical]`, best data mix from #7 | sweep: none/coral/adversarial/domain_norm | best from #7 | RQ4; Level 6 |
| 9 | In-domain (random split) vs LODO cross-site | random split AND LODO, fixed best config from #8 | `[rgb, thermal, electrical]` | best from #8 | best from #7 | **RQ2 — central paper claim**; Level 5 vs 6 comparison |

Every row: minimum 5 seeds (Section 14.3); rows using LODO run k domain rotations x seeds. `ablation_runner.py` is the single entrypoint that executes this entire table (`scripts/run_ablation_matrix.sh`).

**Config <-> row mapping** (which `configs/*.yaml` + CLI overrides produce each row) is documented in `configs/README.md`, generated during Stage 1 (Section 23) and kept in sync by `ablation_runner.py` reading row definitions from `src/evaluation/ablation_matrix.yaml` (a config file listing all 9 rows programmatically, not hard-coded in Python) — this is the actual source of truth; the table above is the human-readable summary of that YAML.


## 16. Reproducibility

- **Seeds:** every training run takes a single `seed` config value; `src/common/seeding.py` seeds Python `random`, `numpy`, `torch` (CPU+CUDA), and sets `torch.backends.cudnn.deterministic=True` / `benchmark=False` when `training.deterministic: true` (config-toggle, since deterministic mode is slower — default `true` for headline runs, may be `false` for hyperparameter search speed).
- **Experiment IDs:** `experiment_id = f"{config_name}__{git_commit[:8]}__{seed}__{timestamp}"`, generated by `src/experiments/registry.py`, appended to `experiments/registry.json` (append-only — never edited in place).
- **Config snapshotting:** the fully-resolved (post-merge, post-CLI-override) config is dumped to `results/<experiment_id>/config.yaml` at run start, before any training happens — this is what "rerun from config" means (Section 24 acceptance criterion), and it must be sufficient on its own (no hidden environment-dependent defaults) to reproduce the run.
- **Git commit recording:** `registry.py` records `git rev-parse HEAD` and `git status --porcelain` (to flag dirty-tree runs) into both the registry and the per-run config snapshot.
- **Package versions:** `pip freeze` output (or `uv pip freeze` if using `uv`) snapshotted to `results/<experiment_id>/environment.txt`.
- **Checkpoint naming:** `checkpoints/<experiment_id>/checkpoint_best.pt` (best val macro-F1) and `checkpoint_last.pt`; checkpoint metadata (epoch, metrics, config hash) stored alongside as `checkpoint_best.meta.json`.
- **Metrics storage:** every run writes `results/<experiment_id>/metrics.json`, `predictions.csv`, `classification_report.json`, `confusion_matrix.png`, `training_curve.png` (Section 17).

---

## 17. Output Artifacts (Fixed Structure)

```text
results/
    <experiment_id>/
        config.yaml                 # fully-resolved config (Section 16)
        environment.txt             # pip freeze snapshot
        metrics.json                # all Section 14.1 metrics
        lodo_results.json           # only present if protocol == lodo
        lodo_summary.json           # only present if protocol == lodo
        confusion_matrix.png
        training_curve.png
        predictions.csv             # sample_id, true_label(s), predicted_label(s), predicted_probs
        classification_report.json  # sklearn-style per-class report
        checkpoint_pointer.txt      # path to the checkpoint in checkpoints/ (checkpoints themselves are NOT in results/)
```

`checkpoints/` and `logs/` are git-ignored (large/regeneratable); `results/` is small and IS kept, since it's what the paper tables/figures are generated from.

---

## 18. Configuration System

**Format:** YAML, loaded via a thin wrapper in `src/common/config.py` around a hierarchical merge (base -> experiment file -> CLI overrides), NOT a heavier framework like Hydra — chosen for simplicity per the "do not overengineer" principle (source blueprint Section 32 analog); `ASSUMPTION`, revisit only if config complexity genuinely outgrows this.

**`configs/base.yaml` contains every key any experiment might need, with defaults; every other config file only overrides what differs.** Full key list (non-exhaustive skeleton — fill exact values during Stage 2/6, Section 23):

```yaml
experiment_name: "unnamed"
seed: 42
n_seeds: 5

data:
  adapters: [irsm, cali, solarfcd, rgbirpv, bimodal_uav, nrel_pvdaq]
  index_path: "data/processed/index/full.parquet"

model:
  modalities: [rgb, thermal, electrical]
  image_backbone: "efficientnet_b0"
  fusion_type: "cross_attention"        # or "late"
  dg_method: "none"                     # none | coral | adversarial | domain_norm
  modality_dropout_prob: 0.15

training:
  loss_type: "focal"                    # or "class_weighted_bce"
  focal_alpha: 0.25
  focal_gamma: 2.0
  coral_weight: 1.0
  adversarial_weight: 1.0
  grl_lambda_schedule: "dann_linear_warmup"
  optimizer: "adamw"
  lr: 1.0e-4
  weight_decay: 1.0e-5
  warmup_steps: 500
  batch_size: 32
  epochs: 50
  deterministic: true
  early_stopping_patience: 10

synthetic:
  ratio: 0.0                            # 0.0 | 0.1 | 0.25 | 0.5 | 1.0 | "unlimited" | "pretrain_finetune" | "domain_randomized"
  ssim_min_threshold: 0.75

evaluation:
  protocol: "random_split"              # or "lodo"
  primary_metric: "macro_f1"

optuna:
  enabled: false
  n_trials: 30

logging:
  tracker: "mlflow"                     # see Section 20
  log_dir: "logs/"

hardware:
  device: "cuda"                        # falls back to cpu automatically if unavailable
  mixed_precision: true
```

**Validation:** `config.py` validates cross-key consistency at load time and fails fast (Section 12's `dg_method != none` requires `protocol: lodo` rule lives here) rather than failing deep inside training after wasted compute.

**A researcher never edits `src/` to run a new experiment** — every experiment is `python -m src.experiments.run --config configs/<file>.yaml [--override key=value ...]`.

---

## 19. Command-Line Interface (Final)

```bash
# One-time data preparation
python -m src.data.download --config configs/base.yaml           # AUTOMATIC datasets only (Section 20)
python -m src.data.indexer --config configs/base.yaml             # builds canonical index, writes DATASET_STATUS.md
python -m src.splitting.lodo_split --config configs/base.yaml     # builds LODO rotation indices

# Synthetic data generation (run once, cached to data/synthetic/)
python -m src.synthetic.electrical_sim --config configs/base.yaml
python -m src.synthetic.image_diffusion --config configs/base.yaml

# Single experiment
python -m src.experiments.run --config configs/baseline_thermal.yaml
python -m src.experiments.run --config configs/synthetic_ablation.yaml --override synthetic.ratio=0.25 seed=1

# Full ablation matrix (Section 15)
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml

# LODO evaluation of an existing/trained config
python -m src.evaluation.lodo_runner --config configs/domain_generalization.yaml --override model.dg_method=coral

# Report generation
python -m src.evaluation.statistical_tests --results_dir results/
python -m src.visualization.ablation_plots --results_dir results/
```

`scripts/*.sh` are one-line wrappers around the above for convenience only (e.g. `scripts/run_ablation_matrix.sh` = the ablation_runner invocation with the standard matrix file) — the Python module invocations above are the canonical interface; the shell scripts must never diverge from them.

---

## 20. Dataset Acquisition

| Dataset | Access mode | Method | Credentials needed | License note |
|---|---|---|---|---|
| InfraredSolarModules | **AUTOMATIC** (if still publicly hosted at build time) | `src/data/download.py` fetches from the Raptor Maps GitHub release | No | Free for research; re-verify current terms at source before relying on it |
| Alfaro-Mejia Cali thermal set | **MANUAL** | paper/repository-hosted; download script prints exact instructions and expected target path `data/raw/cali/` rather than attempting a guessed URL | Possibly (check paper) | Academic — cite required |
| SolarFCD | **MANUAL** (arXiv/dataset-repo release; verify current hosting) | download script prints instructions | Possibly | Open, check dataset repo |
| RGBIRPV | **MANUAL** (journal supplementary / GitHub, verify) | download script prints instructions | Possibly | Check journal supplementary terms |
| Bimodal UAV dataset | **MANUAL** (paper supplementary) | download script prints instructions | Possibly | Check paper supplementary |
| NREL PVDAQ | **MANUAL** (registration required) | download script prints registration URL + expected directory structure once the user has their own export | **Yes — NREL API key / account** | Public with registration |
| ELPV / E-ELPV | **AUTOMATIC** (public GitHub: `zae-bayern/elpv-dataset`) | `src/data/download.py`, but adapter excluded from default registry (Section 6.2) | No | Open, citation required |
| Own-collected MVE-tier data | **MANUAL** (physical collection) | user places files under `data/raw/own_site_<n>/` per the schema documented in `docs/data.md`; `own_site.py` adapter ingests it | n/a | User's own data |

**Hard rule:** `src/data/download.py` never attempts to fabricate or guess a download URL for a MANUAL dataset. For MANUAL entries it prints the exact instructions (source, expected license check, expected target directory) and exits without error — this is not a failure state, it's the correct behavior. `data/DATASET_STATUS.md` (Section 9.2) reflects AUTOMATIC/MANUAL/MISSING/PRESENT status per dataset after every `indexer.py` run, so the researcher always has a truthful current picture.

---

## 21. Hardware Interface (Acquisition Layer, For Future Sensors)

**Module home:** would live at `src/data/acquisition/` if/when implemented — **not built in the initial stages** (Section 23 does not include it; this section exists so the repository doesn't have to be rewritten later). The contract: any future camera/sensor integration produces files that conform to the same `data/raw/<dataset_id>/` structure an `own_site.py`-style adapter expects, and nothing above the adapter layer (preprocessing, splitting, models) ever imports a hardware-specific module. This is already satisfied by the `DatasetAdapter` interface (Section 9.1) — a hardware acquisition script is just another producer of `data/raw/` content, never a special case in the model/training code.


## 22. Environment & Dependencies

- **Python:** 3.11 (`ASSUMPTION`; pin exact patch version in `pyproject.toml` once the build environment is known).
- **PyTorch:** latest stable at build time supporting the target CUDA version; `TO VERIFY` exact pin against the actual GPU/CUDA available (Section 23 Stage 2 confirms and pins this).
- **Single dependency source:** `pyproject.toml` (PEP 621 `[project.dependencies]`); `requirements.txt` is generated FROM `pyproject.toml` for environments that need it (e.g. `pip freeze > requirements.txt` style export), never maintained by hand as a second source of truth. Do not create a separate `environment.yml` unless conda is specifically required by the execution environment (`ASSUMPTION`: not required — pip/venv is sufficient).
- **Core libraries:** `torch`, `timm` (pretrained backbones), `opencv-python`, `albumentations` (augmentation), `diffusers` + `transformers` (diffusion synthesis), `scikit-learn` (classical baselines, metrics, statistical tests), `pvlib` (diode-model physics — mandatory, never hand-roll diode equations per source blueprint), `mlflow` (experiment tracking, Section 20 below — this Section 20 refers to the logging section, not dataset acquisition), `optuna` (hyperparameter search), `pandas`/`pyarrow` (canonical Parquet index), `pyyaml`, `pytest`.
- **OS assumption:** Linux (Ubuntu-class), CUDA-capable GPU optional but recommended (12GB+ VRAM per source blueprint Section 19); code must run on CPU for tests/CI (small fixture data, no GPU required to pass `tests/`).

---

## 23. Testing

### 23.1 Unit tests (`tests/unit/`)
Dataset parsing/adapter correctness (fixture-scale, not full datasets), preprocessing functions (resize/normalize/blur-filter correctness on synthetic tiny images), electrical representation (resampling correctness, scalar-feature extraction against known-analytic diode-model outputs), label mapping (every source class in every `label_maps/*.yaml` maps to one of the 7 canonical classes or `Healthy`, no silent drops), model forward pass (shape checks per Section 9.4, across every `modalities` subset including all missing-modality combinations), fusion (cross-attention vs late fusion output shape + gradient flow), synthetic generator (physical-range validation passes on known-good literature curves, fails on deliberately out-of-range fixtures), split generation (grouped split never puts one `panel_id` in two output sets).

### 23.2 Integration tests (`tests/integration/`)
Raw data -> dataloader (a tiny fixture raw dataset produces a working `DataLoader` end to end through the real adapter + preprocessing + splitting code, not mocks), dataloader -> model (one batch forward pass), training one mini-epoch (loss decreases or at least computes without NaN over a few steps on fixture data), evaluation (metrics.py produces all Section 14.1 fields without error on a small fixture prediction set), checkpoint loading (save -> load -> identical output roundtrip).

### 23.3 Leakage tests (`tests/leakage/`) — explicit, automated, must pass before any headline result is trusted
- `test_no_panel_crosses_split.py`: for every split (random/grouped/site-held-out/LODO), assert no `panel_id` (where non-null) appears in more than one of {train, val, test}.
- `test_no_site_in_train_and_lodo_test.py`: for every LODO rotation, assert the held-out `domain_id` has zero rows in that rotation's train/val index.
- `test_synthetic_not_from_test_split.py`: for every synthetic sample's sidecar metadata (`source_real_sample_id`, Section 11.2) or generation conditioning (Section 11.1's `source_conditioning_domain_id`), assert the source sample was in the training split at generation time — cross-referenced against the split index at test time, not just trusted at generation time (defense in depth).

These three tests are the automated version of the source blueprint's Section 14 leakage table and must run in CI on every commit that touches `src/data/`, `src/splitting/`, or `src/synthetic/`.

---

## 24. Logging + Experiment Tracking

**Decision: MLflow**, local-file-backed (`logs/mlruns/`) by default, no external server dependency required — chosen over W&B for offline/reproducibility-first operation (no account/network dependency for a student/research project) and over bare TensorBoard because MLflow also handles config/param logging and artifact storage in one place (`ASSUMPTION`, config-overridable via `logging.tracker: wandb|tensorboard|mlflow` if the user has W&B access and prefers it — `src/common/logging_setup.py` abstracts this behind one interface so the choice is a config value, not a code change).

**Logged per run:** loss (per step/epoch), all Section 14.1 metrics (per epoch for val, once for test), full resolved hyperparameters (the same content as `results/<experiment_id>/config.yaml`), model architecture summary, dataset version (git commit + index Parquet file hash), confusion matrix image, checkpoint artifact reference (path, not the file itself, to keep MLflow's own storage small — checkpoints live in `checkpoints/`), all `results/<experiment_id>/*.png` plots.

---

## 25. Final Research Tables (Auto-Generated)

**Module home:** `src/evaluation/statistical_tests.py` (computation) + a small table-formatting step in `src/visualization/` or a dedicated `src/evaluation/report_tables.py` (create this file if table formatting logic grows past a few lines — do not overload `statistical_tests.py` with presentation logic).

| Table | Content | Source rows (Section 15) |
|---|---|---|
| 1. Dataset statistics | per-domain sample counts, class balance, modality availability | `data/DATASET_STATUS.md` + canonical index |
| 2. Single-modality performance | rows 1-3 | ablation matrix rows 1,2,3 |
| 3. Multimodal ablation | rows 4,5,6 | ablation matrix rows 4,5,6 |
| 4. Synthetic-data ablation | ratio sweep + pretrain/finetune + domain-randomized | ablation matrix row 7 |
| 5. Cross-site generalization | in-domain vs LODO, generalization gap | ablation matrix row 9 |
| 6. Domain-generalization comparison | ERM vs CORAL vs adversarial vs domain-norm | ablation matrix row 8 |

Each table script reads directly from `results/<experiment_id>/metrics.json` files (never recomputes metrics from raw predictions at table-generation time) and includes bootstrap 95% CIs + significance markers from `statistical_tests.py`.

## 26. Final Figures (Auto-Generated)

`src/visualization/` scripts, one responsibility each (Section 8 module list): system architecture diagram (static, hand-authored once in `docs/architecture.md`, not auto-generated from code), dataset distribution (`confusion_matrix.py`'s sibling utilities or a small `dataset_distribution.py` if this grows), confusion matrices (`confusion_matrix.py`), ROC/PR curves (`roc_pr_curves.py`), cross-domain performance (`domain_gap_plots.py`), synthetic-vs-real comparison (`ablation_plots.py`), ablation plots (`ablation_plots.py`), feature visualization / embedding space (`feature_viz.py`), representative fault examples (manually curated from `data/processed/`, not code-generated — flag as `REQUIRES USER INPUT` at paper-writing time).

---

## 27. Documentation Contents

| File | Must contain |
|---|---|
| `README.md` | project one-paragraph summary, quickstart (env setup -> data prep -> first experiment), link to every `docs/*.md`, link to this spec |
| `docs/architecture.md` | the Section 5 diagram + component table (Section 10), rationale for rejected alternatives (Section 5's "alternatives considered") |
| `docs/data.md` | canonical schema (Section 7), how to add a new dataset adapter, expected `data/raw/` structure per dataset |
| `docs/datasets.md` | the Section 6.2/20 acquisition table, license notes, AUTOMATIC/MANUAL/OPTIONAL status |
| `docs/training.md` | pipeline stages (Section 13.1), how to launch a run, config key reference (points to Section 18) |
| `docs/evaluation.md` | metrics definitions (Section 14.1), LODO protocol, how to interpret generalization gap |
| `docs/synthetic_data.md` | Section 11 in full, including the fault-injection table location and how to extend it |
| `docs/domain_generalization.md` | Section 12 method comparison table, how to add a new DG method |

---

## 28. ANTIGRAVITY BUILD PLAN

Each stage lists: objective, files to create/modify, dependencies, implementation requirements, tests, expected output, acceptance criteria, and what must work before proceeding. **Do not start a stage until the previous stage's acceptance criteria pass.**

### Stage 1 — Repository scaffolding
- **Objective:** create the full directory tree (Section 8) with placeholder `__init__.py`/`.gitkeep` files; no logic yet.
- **Files:** entire tree in Section 8.
- **Dependencies:** none.
- **Tests:** `tests/` directory exists with empty placeholder test files that import nothing yet (so `pytest` runs green trivially).
- **Acceptance:** `find pv_fault_detection -type d` matches Section 8; `git init` + first commit done; `.gitignore` correctly excludes `data/`, `checkpoints/`, `logs/`.

### Stage 2 — Environment setup
- **Objective:** working, pinned Python environment.
- **Files:** `pyproject.toml`, `.env.example`.
- **Implementation:** pin Python + core libraries (Section 22); confirm CUDA/PyTorch compatibility against the actual build machine (`TO VERIFY` at this stage, not before).
- **Tests:** `pip install -e .` (or `uv sync`) succeeds; `python -c "import torch, timm, pvlib, mlflow, optuna, albumentations, diffusers"` succeeds.
- **Acceptance:** clean environment installs without conflicts; record exact resolved versions in `results/` is not applicable yet — just confirm install works.

### Stage 3 — Dataset infrastructure
- **Objective:** `DatasetAdapter` interface + at least 2 real adapters (start with the AUTOMATIC ones: InfraredSolarModules, ELPV) working end to end against real downloaded data.
- **Files:** `src/data/adapters/base.py`, `irsm.py`, `elpv.py`, `download.py`.
- **Dependencies:** Stage 2.
- **Tests:** `tests/unit/test_adapters.py` for these two adapters against small fixtures.
- **Acceptance:** `python -m src.data.download` fetches IRSM automatically; `iter_samples()`/`to_canonical()` run on real downloaded data without error.

### Stage 4 — Canonical data model
- **Objective:** all remaining adapters (MANUAL ones get instruction-printing stubs until the user supplies data) + the indexer producing one valid canonical Parquet index.
- **Files:** remaining `src/data/adapters/*.py`, `src/data/indexer.py`, `src/data/validate.py`, `label_maps/*.yaml`.
- **Tests:** `tests/unit/test_adapters.py` (all adapters), `tests/unit/test_label_mapping.py`.
- **Acceptance:** `python -m src.data.indexer` produces `data/processed/index/full.parquet` conforming exactly to Section 7 schema (validated by `validate.py`); `data/DATASET_STATUS.md` correctly reports AUTOMATIC/MANUAL/PRESENT/MISSING per dataset.

### Stage 5 — Preprocessing
- **Objective:** all `src/preprocessing/*` modules working against real indexed data.
- **Files:** `rgb.py`, `thermal.py`, `electrical.py`, `environment.py`, `synchronization.py`, `quality_control.py`.
- **Tests:** `tests/unit/test_preprocessing.py`.
- **Acceptance:** processing the full canonical index produces `data/processed/images|electrical/` outputs without error; quality-control filtering rate is logged and reasonable (not near-100% rejection, which would indicate a bug).

### Stage 6 — Splitting
- **Objective:** `src/splitting/*` implemented and leakage-tested.
- **Files:** `random_split.py`, `grouped_split.py`, `site_held_out_split.py`, `lodo_split.py`.
- **Tests:** `tests/leakage/test_no_panel_crosses_split.py`, `test_no_site_in_train_and_lodo_test.py`.
- **Acceptance:** all leakage tests pass on real indexed data, not just fixtures.

### Stage 7 — Baseline model (single modality)
- **Objective:** `src/models/encoders/image_encoder.py`, `electrical_encoder.py`, minimal `pv_fault_model.py` supporting single-modality forward pass, `heads/classifier_head.py`.
- **Tests:** `tests/unit/test_model_forward.py` (single-modality subset only at this stage).
- **Acceptance:** forward pass on real preprocessed batch produces correctly-shaped logits; no NaNs.

### Stage 8 — Training pipeline
- **Objective:** `src/training/*` (trainer, losses, optim), `configs/base.yaml` + `configs/baseline_thermal.yaml`, `src/experiments/run.py`, `registry.py`.
- **Tests:** `tests/integration/test_train_one_step.py`, `test_checkpoint_roundtrip.py`.
- **Acceptance:** `python -m src.experiments.run --config configs/baseline_thermal.yaml` completes a full (small-epoch-count) run end to end, writes all Section 17 output artifacts, logs to MLflow.

### Stage 9 — Evaluation
- **Objective:** `src/evaluation/metrics.py`, basic `random_split` evaluation wired into `run.py`.
- **Tests:** `tests/integration/test_evaluate.py`.
- **Acceptance:** `results/<experiment_id>/metrics.json` contains every Section 14.1 field with sane values on the Stage 8 run.

### Stage 10 — Multimodal architecture
- **Objective:** `fusion/cross_attention.py`, `fusion/late_fusion.py`, full `pv_fault_model.py` (all modality combinations + missing-modality handling), `configs/multimodal_rgb_thermal.yaml`, `multimodal_full.yaml`.
- **Tests:** `tests/unit/test_fusion.py`, full `tests/unit/test_model_forward.py` (all modality subsets).
- **Acceptance:** ablation matrix rows 1-6 (Section 15) each run successfully end to end.

### Stage 11 — Synthetic subsystem
- **Objective:** `src/synthetic/*` complete; `data/synthetic/` populated; `configs/synthetic_ablation.yaml`.
- **Tests:** `tests/unit/test_synthetic_generator.py`, `tests/leakage/test_synthetic_not_from_test_split.py`.
- **Acceptance:** generated electrical curves pass physical-range validation at a reasonable acceptance rate; ablation matrix row 7 runs across the full ratio sweep.

### Stage 12 — Domain generalization
- **Objective:** `src/models/domain_generalization/*`, `src/evaluation/lodo_runner.py`, `configs/domain_generalization.yaml`, `lodo_final.yaml`.
- **Tests:** `tests/leakage/test_no_site_in_train_and_lodo_test.py` (full real-data run, not just fixture), `tests/unit/test_model_forward.py` extended for DG-head presence.
- **Acceptance:** ablation matrix rows 8-9 run; ERM-vs-DG comparison produces a real (possibly null) result.

### Stage 13 — Experiment automation
- **Objective:** `src/evaluation/ablation_runner.py`, `ablation_matrix.yaml`, `src/evaluation/statistical_tests.py`, `scripts/*.sh`.
- **Acceptance:** `python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml` runs the entire 9-row matrix unattended (given adequate compute time) and produces the full `results/` tree with CIs and significance tests.

### Stage 14 — Testing completeness
- **Objective:** fill any remaining gaps in `tests/unit/`, `tests/integration/`, `tests/leakage/` so every module in Section 9/Section 8 has coverage.
- **Acceptance:** `pytest tests/` fully green, including all leakage tests, on CPU, in under a reasonable CI time budget (fixture-scale data only for unit/integration; leakage tests may run against real indexed data and take longer — mark accordingly with `pytest.mark.slow`).

### Stage 15 — Documentation & final validation
- **Objective:** all `docs/*.md` written to the Section 27 spec; `src/visualization/*` complete; `src/evaluation/report_tables.py` generating Section 25's 6 tables.
- **Acceptance:** Section 24 (below) acceptance criteria fully pass; `STATUS.md` reflects a complete build.


## 29. Acceptance Criteria (Definition of "Project Completed")

### Data layer
- [ ] All AUTOMATIC datasets can be indexed without manual intervention
- [ ] Canonical metadata generated conforming exactly to Section 7 schema
- [ ] Leakage checks pass on real (not just fixture) data
- [ ] `data/DATASET_STATUS.md` accurately reflects AUTOMATIC/MANUAL/PRESENT/MISSING status

### Model layer
- [ ] All 3 single-modality baselines train (rows 1-3)
- [ ] Full multimodal model trains (row 5)
- [ ] Missing-modality handling works (verified across all modality subsets in `test_model_forward.py`)

### Synthetic layer
- [ ] Synthetic generation works for both electrical and image modalities
- [ ] Synthetic data can be mixed at every configured ratio
- [ ] Synthetic-vs-real experiment (row 7) runs and produces a generalization-gap comparison, not just an in-domain accuracy comparison

### Domain generalization
- [ ] LODO split works with real leakage-test verification
- [ ] All 4 DG configurations (none/coral/adversarial/domain_norm) train
- [ ] Unseen-site evaluation works and produces a real (possibly null) ERM-vs-DG comparison

### Research layer
- [ ] All 9 ablation-matrix rows are reproducible from their config alone
- [ ] All Section 14.1/14.3 metrics generated with CIs and significance tests
- [ ] All 6 Section 25 tables generated automatically from `results/`
- [ ] All Section 26 figures generated automatically

### Engineering layer
- [ ] `pytest tests/` passes, including all leakage tests
- [ ] All `docs/*.md` complete per Section 27
- [ ] Environment reproducible from `pyproject.toml` alone
- [ ] One-command experiment execution (`python -m src.experiments.run --config ...`) and one-command full-matrix execution both work

**Explicit non-goals (do not build):** a 500-file enterprise architecture, a shared multimodal transformer tokenizer (rejected in Section 5), spectrogram representations of electrical curves (rejected — I-V curves are not periodic signals), a full digital-twin infrastructure (only the electrical-simulation component is in scope), self-supervised pretraining on massive unlabeled imagery (explicitly out of scope per source blueprint Section 9 — note as future work in `docs/domain_generalization.md` instead of building it).

---

## 30. Final End-to-End Workflow

```bash
# 1. Environment
pip install -e .

# 2. Data acquisition (AUTOMATIC datasets fetched; MANUAL ones print instructions)
python -m src.data.download --config configs/base.yaml
# -> user manually places MANUAL-dataset files under data/raw/<dataset_id>/ per docs/datasets.md, then re-runs:
python -m src.data.download --config configs/base.yaml

# 3. Canonical indexing
python -m src.data.indexer --config configs/base.yaml
cat data/DATASET_STATUS.md   # confirm expected datasets are PRESENT

# 4. Preprocessing (invoked internally by indexer/splitting, or explicitly if a standalone entrypoint is added)
python -m src.splitting.lodo_split --config configs/base.yaml

# 5. Synthetic data generation (once, cached)
python -m src.synthetic.electrical_sim --config configs/base.yaml
python -m src.synthetic.image_diffusion --config configs/base.yaml

# 6. Research ladder, one level at a time, cheapest first
python -m src.experiments.run --config configs/baseline_thermal.yaml
python -m src.experiments.run --config configs/baseline_rgb.yaml
python -m src.experiments.run --config configs/baseline_electrical.yaml
python -m src.experiments.run --config configs/multimodal_rgb_thermal.yaml
python -m src.experiments.run --config configs/multimodal_full.yaml

# 7. Full ablation matrix (Section 15), unattended
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml

# 8. Reports
python -m src.evaluation.statistical_tests --results_dir results/
python -m src.evaluation.report_tables --results_dir results/ --out reports/
python -m src.visualization.ablation_plots --results_dir results/ --out reports/figures/
```

At every step, `STATUS.md` and `data/DATASET_STATUS.md` give a truthful current picture of what's built/populated — Antigravity updates `STATUS.md` at the end of each Section 28 stage, never claiming a stage complete if its acceptance criteria didn't pass.

---

## Appendix — Decision Classification Key

Every non-obvious decision above is one of:
- **DECIDED** — final, taken directly from the source research blueprint, do not re-derive.
- **ASSUMPTION** — a reasonable default filled in because the source document didn't specify it; config-overridable; flagged inline above.
- **TO VERIFY** — must be checked against a primary source (dataset license/availability, literature fault-parameter ranges) before relying on it in a real run; flagged inline above.
- **REQUIRES USER INPUT** — genuinely cannot be decided by the coding agent (e.g. actual GPU/CUDA version on the build machine, real-site data collection logistics, representative fault-example figure curation); flagged inline above.

Do not silently convert an ASSUMPTION/TO VERIFY/REQUIRES USER INPUT item into a DECIDED one without flagging it in `STATUS.md`.
