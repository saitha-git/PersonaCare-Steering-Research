#!/usr/bin/env bash
# Run baseline / alpha_0 / alpha_4 Qwen3-1.7B Genie bundles with one prompt.
#
# This must run on a machine/device with the Genie SDK executable available.
# The current dev machine may not have `genie-t2t-run` on PATH; set GENIE_BIN
# to its absolute path if needed.
#
# Usage:
#   GENIE_BIN=/path/to/genie-t2t-run bash scripts/run_qwen3_1_7b_genie_compare.sh
#   PROMPT="What is gravity? Keep the answer under ten words." bash scripts/run_qwen3_1_7b_genie_compare.sh
set -euo pipefail

cd "$(dirname "$0")/.."

GENIE_BIN="${GENIE_BIN:-genie-t2t-run}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3_1_7b_patch}"
BUNDLE_NAME="${BUNDLE_NAME:-qwen3_1_7b-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_for_galaxy}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/genie_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
PROMPT="${PROMPT:-What is gravity? Keep the answer under ten words.}"
PHASES="${PHASES:-baseline alpha_0 alpha_4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if ! command -v "$GENIE_BIN" >/dev/null 2>&1 && [ ! -x "$GENIE_BIN" ]; then
    cat >&2 <<EOF
Genie runner not found: $GENIE_BIN

Install the Genie SDK on the target device, or set GENIE_BIN to the absolute
path of genie-t2t-run. The generated bundles are under:
  $OUT_ROOT/{baseline,alpha_0,alpha_4}/$BUNDLE_NAME
EOF
    exit 2
fi

mkdir -p "$RUN_DIR"
printf "%s\n" "$RUN_DIR" > "$OUT_ROOT/latest_genie_run_dir.txt"

summary="$RUN_DIR/summary.jsonl"
: > "$summary"

echo "Run dir: $RUN_DIR"
echo "Prompt:  $PROMPT"
echo "Phases:  $PHASES"

for phase in $PHASES; do
    bundle="$OUT_ROOT/$phase/$BUNDLE_NAME"
    if [ ! -f "$bundle/genie_config.json" ]; then
        echo "Missing $bundle/genie_config.json" >&2
        exit 2
    fi

    log="$RUN_DIR/$phase.log"
    echo "==== $phase ===="
    echo "Bundle: $bundle"
    echo "Log:    $log"

    set +e
    (
        cd "$bundle"
        "$GENIE_BIN" -c genie_config.json -p "$PROMPT" $EXTRA_ARGS
    ) >"$log" 2>&1
    rc=$?
    set -e

    PHASE="$phase" \
    RC="$rc" \
    BUNDLE="$bundle" \
    LOG="$log" \
    PROMPT_TEXT="$PROMPT" \
    EXTRA_ARGS_TEXT="$EXTRA_ARGS" \
    SUMMARY="$summary" \
    python - <<'PY'
import json
import os
from pathlib import Path
phase = os.environ["PHASE"]
log = Path(os.environ["LOG"])
text = log.read_text(encoding="utf-8", errors="replace")
print(json.dumps({
    "phase": phase,
    "returncode": int(os.environ["RC"]),
    "bundle": os.environ["BUNDLE"],
    "log": str(log),
    "prompt": os.environ["PROMPT_TEXT"],
    "extra_args": os.environ["EXTRA_ARGS_TEXT"],
    "last_20_lines": text.splitlines()[-20:],
}, ensure_ascii=False), file=open(os.environ["SUMMARY"], "a", encoding="utf-8"))
PY

    tail -40 "$log" || true
    if [ "$rc" -ne 0 ]; then
        echo "$phase failed with rc=$rc" >&2
        exit "$rc"
    fi
done

python - <<PY
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("$summary").read_text(encoding="utf-8").splitlines()]
Path("$RUN_DIR/summary.json").write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
PY

echo
echo "Saved logs -> $RUN_DIR"
echo "Compare baseline vs alpha_0 vs alpha_4 in:"
echo "  $RUN_DIR/summary.json"
