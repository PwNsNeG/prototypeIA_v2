#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

python encode_corpus.py \
  --train-jsonl data/prepared/corpus_v1/train.jsonl \
  --val-jsonl data/prepared/corpus_v1/val.jsonl \
  --tokenizer data/tokenizer/bpe16k/tokenizer.json \
  --out-dir data/tokenized/corpus_v1_bpe16k_bs512  \
  --block-size 512 \
  --append-eos
