#!/usr/bin/env bash
# scripts/run_experiment.sh
# Thin wrapper: run a single experiment.
set -euo pipefail
CONFIG=${1:-"configs/baseline_thermal.yaml"}
shift || true
python -m src.experiments.run --config "$CONFIG" "$@"
