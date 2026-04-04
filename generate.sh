#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh


python3 generate.py \
  --checkpoint checkpoints/best.pt \
  --tokenizer data/tokenizer/bpe16k/tokenizer.json
