#!/usr/bin/env bash
# scripts/prepare_data.sh
# Thin wrapper: download AUTOMATIC datasets + build canonical index.
set -euo pipefail
CONFIG=${1:-"configs/base.yaml"}
python -m src.data.download --config "$CONFIG"
python -m src.data.indexer --config "$CONFIG"
