#!/usr/bin/env bash
# scripts/generate_report_tables.sh
# Thin wrapper: generate all research tables and figures.
set -euo pipefail
python -m src.evaluation.statistical_tests --results_dir results/
python -m src.evaluation.report_tables --results_dir results/ --out reports/
python -m src.visualization.ablation_plots --results_dir results/ --out reports/figures/
