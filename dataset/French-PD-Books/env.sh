#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="requirements.txt"

echo "[*] Using Python: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[!] Python not found: $PYTHON_BIN"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[*] Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

if [ ! -f "$REQ_FILE" ]; then
  echo "[*] No $REQ_FILE found, creating a minimal one"
  cat > "$REQ_FILE" <<'EOF'
pyarrow
numpy
EOF
fi

echo "[*] Installing requirements from $REQ_FILE"
pip install -r "$REQ_FILE"

echo
echo "[+] Environment ready."
echo "[+] Virtualenv: $VENV_DIR"
echo "[+] Python: $(which python)"
echo "[+] Installed packages:"
pip list | grep -E 'pyarrow|numpy' || true
