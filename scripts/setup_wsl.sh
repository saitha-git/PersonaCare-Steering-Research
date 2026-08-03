#!/usr/bin/env bash
# Setup for Ubuntu under WSL2 (tested target: Ubuntu 22.04/24.04, Python 3.10-3.12).
# The main POC also runs natively on Windows; this path exists because Qualcomm's
# QAIRT SDK tooling (qairt-converter, qnn-net-run x86 emulation) is Linux-first.
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install --upgrade pip

# CUDA build if an NVIDIA GPU is visible from WSL, else CPU build.
if command -v nvidia-smi >/dev/null 2>&1; then
    pip install torch --index-url https://download.pytorch.org/whl/cu128
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
pip install -e . --no-deps

python -m pytest tests -q
python -m steering_poc.qualcomm.detect_environment
echo "WSL environment ready. Activate with: source .venv-wsl/bin/activate"
