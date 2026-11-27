#!/usr/bin/env bash
# scripts/run_ablation_matrix.sh
# Thin wrapper: run the full 9-row ablation matrix.
set -euo pipefail
python -m src.evaluation.ablation_runner --matrix src/evaluation/ablation_matrix.yaml "$@"
