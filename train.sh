#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

python train.py \
  --data-dir data/tokenized/corpus_v1_bpe16k_bs512 \
  --tokenizer data/tokenizer/bpe16k/tokenizer.json \
  --checkpoints checkpoints \
  --batch-size 16 \
  --max-iters 40000 \
  --eval-interval 50 \
  --eval-iters 20 \
  --learning-rate 3e-4 \
  --warmup-steps 100 \
  --n-embd 384 \
  --n-head 6 \
  --n-layer 6 \
  --dropout 0.1 \
  --sample-prompt "Monsieur,"
