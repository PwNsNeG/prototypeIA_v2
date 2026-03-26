#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ./env.sh
python build_corpus.py \
  --data-dir ./dataset/French-PD-Books \
  --glob "gallica_mono_*.parquet" \
  --out-dir ./data/prepared/corpus_v1 \
  --target-chars 500000000 \
  --min-chars 2500 \
  --min-alpha-ratio 0.58 \
  --max-weird-ratio 0.06 \
  --min-score 1.5 \
  --val-ratio 0.01 \
  --write-rejected
