#!/usr/bin/env bash
set -euo pipefail
sudo apt-get install pip python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

