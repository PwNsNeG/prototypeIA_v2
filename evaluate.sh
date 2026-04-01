#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

python evaluate.py \
  --ckpt checkpoints/best.pt \
  --tokenizer data/tokenizer/bpe16k/tokenizer.json \
  --data-dir data/tokenized/corpus_v1_bpe16k_bs512
