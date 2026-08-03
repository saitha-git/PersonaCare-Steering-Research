#!/usr/bin/env bash
# Dedicated Python 3.10 environment in WSL for Qualcomm tooling
# (qai-hub client, ONNX tooling, and — when installed — the QAIRT SDK's Linux
# x86-64 binaries, which are Linux-first and historically target Python 3.10).
# This is SEPARATE from the model/eval environment (Windows .venv, Python 3.13):
# steering/eval runs there; Qualcomm CLI work runs here.
set -euo pipefail

VENV="$HOME/.venvs/qualcomm310"

# uv provides a standalone CPython 3.10 without touching system Python.
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.10
uv venv --python 3.10 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install --python "$VENV/bin/python" \
    "qai-hub==0.52.0" onnx onnxruntime numpy

python - <<'EOF'
import sys, onnx, onnxruntime, qai_hub
print("python", sys.version.split()[0])
print("qai-hub", qai_hub.__version__)
print("onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__)
EOF

cat <<'EOF'

Qualcomm WSL env ready: source ~/.venvs/qualcomm310/bin/activate
Next steps (manual, license acceptance required):
  * Configure AI Hub:  qai-hub configure --api_token <token>
  * QAIRT SDK (Linux): download via Qualcomm Package Manager / qpm-cli, then
    export QNN_SDK_ROOT=<install>/qairt/<version>
    and run: python -m steering_poc.qualcomm.qnn_emulation --run
EOF
