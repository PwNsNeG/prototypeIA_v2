#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh


python3 export_model.py \
  --checkpoint checkpoints/best.pt \
  --out-dir export/best \
  --copy-model-py
