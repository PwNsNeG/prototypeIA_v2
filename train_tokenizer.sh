#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

python train_tokenizer.py \
  --input data/prepared/corpus_v1/train.jsonl \
  --out-dir data/tokenizer/bpe16k \
  --vocab-size 16000 \
  --min-frequency 2
