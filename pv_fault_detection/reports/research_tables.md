# PV Fault Detection Research Report — Auto-Generated Tables

> Generated from 1 completed experiment runs in `results`.
> Rows marked `NOT_YET_RUN` require the corresponding experiment to be executed first.
> **No values in this report are fabricated or hardcoded.**

## Table1 Single Modality

| Experiment                 | Macro-F1    | Balanced Accuracy   | ECE         | Hotspot Recall   | Params      | Latency (ms/sample)   | Source                                       |
|:---------------------------|:------------|:--------------------|:------------|:-----------------|:------------|:----------------------|:---------------------------------------------|
| RGB-only (Baseline)        | NOT_YET_RUN | NOT_YET_RUN         | NOT_YET_RUN | NOT_YET_RUN      | NOT_YET_RUN | NOT_YET_RUN           | NOT_YET_RUN                                  |
| Thermal-only (Baseline)    | NOT_YET_RUN | NOT_YET_RUN         | NOT_YET_RUN | NOT_YET_RUN      | NOT_YET_RUN | NOT_YET_RUN           | baseline_thermal__nogit__42__20260830T135617 |
| Electrical-only (Baseline) | NOT_YET_RUN | NOT_YET_RUN         | NOT_YET_RUN | NOT_YET_RUN      | NOT_YET_RUN | NOT_YET_RUN           | NOT_YET_RUN                                  |

## Table2 Multimodal Ablation

| Experiment                       | Macro-F1    | ECE         | Source      |
|:---------------------------------|:------------|:------------|:------------|
| RGB + Thermal (Bimodal)          | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Full Trimodal (RGB+Thermal+Elec) | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Late Fusion vs Cross-Attention   | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |

## Table3 Synthetic Sweep

| Synthetic Ratio   | Macro-F1    | ECE         | Source      |
|:------------------|:------------|:------------|:------------|
| 0.0               | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| 0.1               | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| 0.25              | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| 0.5               | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| 1.0               | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| unlimited         | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| pretrain_finetune | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| domain_randomized | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |

## Table4 Generalization Gap

| Evaluation Protocol         | Macro-F1    | Std-F1           | Source      |
|:----------------------------|:------------|:-----------------|:------------|
| In-Domain (Grouped Split)   | NOT_YET_RUN | N/A (single run) | NOT_YET_RUN |
| Cross-Site LODO (Zero-Shot) | NOT_YET_RUN | NOT_YET_RUN      | NOT_YET_RUN |

## Table5 Dg Methods

| DG Method   | LODO Mean Macro-F1   | LODO Std Macro-F1   | Source      |
|:------------|:---------------------|:--------------------|:------------|
| none        | NOT_YET_RUN          | NOT_YET_RUN         | NOT_YET_RUN |
| coral       | NOT_YET_RUN          | NOT_YET_RUN         | NOT_YET_RUN |
| adversarial | NOT_YET_RUN          | NOT_YET_RUN         | NOT_YET_RUN |
| domain_norm | NOT_YET_RUN          | NOT_YET_RUN         | NOT_YET_RUN |

## Table6 Per Class Breakdown

| Fault Class            | Precision   | Recall      | F1-Score    | Source      |
|:-----------------------|:------------|:------------|:------------|:------------|
| Healthy                | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Hotspot                | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Open/Short-circuit     | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Shading/Obstruction    | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Soiling                | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Structural (visual)    | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |
| Electrical/Diode fault | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN | NOT_YET_RUN |

