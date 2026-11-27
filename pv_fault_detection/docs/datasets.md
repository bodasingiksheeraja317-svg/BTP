# Datasets

See `data/DATASET_STATUS.md` for current acquisition status.

## Dataset Table

| Dataset | Domain ID | Modality | Access | License | Notes |
|---|---|---|---|---|---|
| InfraredSolarModules (Raptor Maps) | irsm | Thermal | AUTOMATIC | Free for research — TO VERIFY | |
| Alfaro-Mejia Cali thermal | cali | Thermal | MANUAL | Academic — TO VERIFY | |
| SolarFCD | solarfcd | RGB+Thermal | MANUAL | Open — TO VERIFY | |
| RGBIRPV | rgbirpv | RGB+IR paired | MANUAL | TO VERIFY | |
| Bimodal UAV (YOLOv11-MO) | bimodal_uav | RGB+Thermal | MANUAL | TO VERIFY | |
| NREL PVDAQ | nrel_pvdaq | Electrical+env | MANUAL (registration) | Public with registration | REQUIRES NREL account |
| ELPV | elpv | EL | AUTOMATIC | Open, cite Deitsch 2019 | Optional sub-study only |
| Own site | own_site_* | RGB+Thermal | User-collected | User's own data | |

## Download Instructions

Run `python -m src.data.download --config configs/base.yaml` for AUTOMATIC datasets.
Manual datasets print detailed instructions when you run the above command.

See `data/DATASET_STATUS.md` after running for current status.
