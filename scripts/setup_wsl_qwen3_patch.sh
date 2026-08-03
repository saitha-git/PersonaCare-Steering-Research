#!/usr/bin/env bash
# WSL/Linux setup for the Qwen3-1.7B steering-inside-qai-hub-models experiment.
#
# This creates a Python 3.10 env, clones qai-hub-models into an ignored
# external/ checkout, and installs both this repo and qai-hub-models editable.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${VENV:-$HOME/.venvs/qwen3_patch310}"
QHM_DIR="${QHM_DIR:-external/qai-hub-models}"
QHM_REPO="${QHM_REPO:-https://github.com/qualcomm/ai-hub-models.git}"
QHM_REF="${QHM_REF:-main}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.10
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python 3.10 "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

if [ ! -d "$QHM_DIR/.git" ]; then
    mkdir -p "$(dirname "$QHM_DIR")"
    git clone "$QHM_REPO" "$QHM_DIR"
fi
git -C "$QHM_DIR" fetch --all --tags
git -C "$QHM_DIR" checkout "$QHM_REF"

uv pip install --python "$VENV/bin/python" --upgrade pip
uv pip install --python "$VENV/bin/python" -r "$QHM_DIR/src/qai_hub_models/requirements.txt"
uv pip install --python "$VENV/bin/python" -r "$QHM_DIR/src/qai_hub_models/models/qwen3_1_7b/requirements.txt"
uv pip install --python "$VENV/bin/python" -e "$QHM_DIR/cli" --no-deps
uv pip install --python "$VENV/bin/python" -e "$QHM_DIR/src" --no-deps
uv pip install --python "$VENV/bin/python" -e . --no-deps

python - <<'EOF'
import sys
import torch
import transformers
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
try:
    import qai_hub_models
    print("qai_hub_models", getattr(qai_hub_models, "__version__", "editable"))
except Exception as exc:
    print("qai_hub_models import failed:", exc)
EOF

cat <<EOF

Qwen3 patch env ready.
Activate it with:
  source "$VENV/bin/activate"

Next:
  bash scripts/run_qwen3_1_7b_patch_experiment.sh

Optional:
  qai-hub configure --api_token <token>
EOF
