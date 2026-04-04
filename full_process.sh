#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="data/tokenized/corpus_v1_bpe16k_bs512"

run_step() {
  local name="$1"
  local script="$2"

  echo
  echo "============================================================"
  echo "[*] ${name}"
  echo "============================================================"

  if [[ ! -f "$script" ]]; then
    echo "[!] Missing script: $script"
    exit 1
  fi

  bash "$script"

  echo "[+] Completed: ${name}"
}

echo "[*] Starting full pipeline"

# 1. Environment
run_step "Prepare environment" "./env.sh"

# 2. Build corpus
run_step "Build corpus" "./build_corpus.sh"

# 3. Train tokenizer
run_step "Train tokenizer" "./train_tokenizer.sh"

# 4. Encode corpus
run_step "Encode corpus" "./encode_corpus.sh"

# 5. Sanity check (important)
echo
echo "[*] Checking encoded data directory..."
if [[ ! -d "$DATA_DIR" ]]; then
  echo "[!] Expected data directory not found: $DATA_DIR"
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.bin" ]]; then
  echo "[!] Missing train.bin in $DATA_DIR"
  exit 1
fi

echo "[+] Data directory OK: $DATA_DIR"

# 6. Train model (force correct path)
echo
echo "[*] Training model..."
run_step "Training" "./train.sh"


echo "[+] Training completed"

# 7. Evaluate model (force correct path)
echo
echo "[*] Evaluating model..."

run_step "Evaluate the model" "./evaluate.sh"

echo "[+] Evaluation completed"

echo
echo "[+] Full process completed successfully."

# 8. export model 
echo
echo "[*] Export model..."

run_step "Export the model" "./export_model.sh"

echo
echo "[+] Full process completed successfully."
# 9.  Generate model
echo
echo "[*] Generate model..."

run_step "Generate the standalone model" "./generate.sh"

echo
echo "[+] Full process completed successfully."

