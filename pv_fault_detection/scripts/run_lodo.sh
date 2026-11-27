#!/usr/bin/env bash
# scripts/run_lodo.sh
# Thin wrapper: run LODO evaluation for a config.
set -euo pipefail
CONFIG=${1:-"configs/lodo_final.yaml"}
shift || true
python -m src.evaluation.lodo_runner --config "$CONFIG" "$@"
