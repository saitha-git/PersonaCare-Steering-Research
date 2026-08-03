#!/usr/bin/env bash
# Run one Qwen3-1.7B steering-patch export phase with durable logs.
#
# Examples:
#   bash scripts/run_qwen3_1_7b_patch_step.sh vector
#   bash scripts/run_qwen3_1_7b_patch_step.sh baseline
#   bash scripts/run_qwen3_1_7b_patch_step.sh alpha4
#   bash scripts/run_qwen3_1_7b_patch_step.sh all
#
# Optional knobs:
#   SEQUENCE_LENGTHS=1,128 CONTEXT_LENGTHS=512 DEVICE="Samsung Galaxy S25 (Family)"
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE="${1:-baseline}"
QHM_DIR="${QHM_DIR:-external/qai-hub-models}"
PATCH_FILE="${PATCH_FILE:-patches/qai_hub_models_qwen3_1_7b_steering.patch}"
CONFIG="${CONFIG:-configs/qwen3_1_7b.yaml}"
DATA="${DATA:-data/contrast_pairs.jsonl}"
LAYER="${LAYER:-7}"
TAG="${TAG:-qwen3_1_7b}"
VECTOR="${VECTOR:-artifacts/vector_layer_${LAYER}_${TAG}.pt}"
DEVICE="${DEVICE:-Samsung Galaxy S25 (Family)}"
RUNTIME="${RUNTIME:-geniex_qairt}"
EXPORT_MODULE="${EXPORT_MODULE:-qai_hub_models.models.qwen3_1_7b.export}"
OUT_ROOT="${OUT_ROOT:-artifacts/qwen3_1_7b_patch}"
SEQUENCE_LENGTHS="${SEQUENCE_LENGTHS:-1,128}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-512}"
RUN_ROOT="${RUN_ROOT:-$OUT_ROOT/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$PHASE}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
LOG="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"
printf "%s\n" "$RUN_DIR" > "$OUT_ROOT/latest_run_dir.txt"
printf "%s\n" "$$" > "$RUN_DIR/runner.pid"

exec > >(tee -a "$LOG") 2>&1

log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_qhm() {
    if [ ! -d "$QHM_DIR/.git" ]; then
        log "$QHM_DIR missing; run scripts/setup_wsl_qwen3_patch.sh first."
        exit 2
    fi
}

apply_patch_if_needed() {
    require_qhm
    local model_py="$QHM_DIR/src/qai_hub_models/models/_shared/qwen3/model.py"
    if ! grep -q "STEERING_POC_QWEN3_VECTOR" "$model_py"; then
        log "Applying $PATCH_FILE to $QHM_DIR"
        git -C "$QHM_DIR" apply "$(pwd)/$PATCH_FILE"
    else
        log "Patch already present in $model_py"
    fi
}

ensure_vector() {
    mkdir -p "$(dirname "$VECTOR")"
    if [ ! -f "$VECTOR" ]; then
        log "Extracting Qwen3-1.7B steering vector -> $VECTOR"
        PYTHONUNBUFFERED=1 python -m steering_poc.extract \
            --config "$CONFIG" \
            --data "$DATA" \
            --layers "$LAYER" \
            --tag "$TAG"
    else
        log "Using existing vector $VECTOR"
    fi

    python - <<EOF
import torch
payload = torch.load("$VECTOR", map_location="cpu")
vector = payload["vector"] if isinstance(payload, dict) else payload
print("vector", "$VECTOR", "shape", tuple(vector.shape), "norm", float(vector.float().norm()))
assert vector.numel() == 2048, "Qwen3-1.7B steering vector must be 2048-dim"
EOF
}

run_export() {
    local label="$1"
    local alpha="${2:-}"
    local out_dir="$OUT_ROOT/$label"
    local status="$RUN_DIR/$label.status"

    mkdir -p "$out_dir"
    log "Starting export label=$label out=$out_dir"
    log "runtime=$RUNTIME device=$DEVICE sequence_lengths=$SEQUENCE_LENGTHS context_lengths=$CONTEXT_LENGTHS"
    printf "running\n" > "$status"

    set +e
    if [ "$label" = "baseline" ]; then
        env -u STEERING_POC_QWEN3_VECTOR \
            -u STEERING_POC_QWEN3_LAYER \
            -u STEERING_POC_QWEN3_ALPHA \
            -u STEERING_POC_QWEN3_NORMALIZE \
            PYTHONUNBUFFERED=1 \
            python -m "$EXPORT_MODULE" \
                --device "$DEVICE" \
                --runtime "$RUNTIME" \
                --sequence-lengths "$SEQUENCE_LENGTHS" \
                --context-lengths "$CONTEXT_LENGTHS" \
                --skip-profiling \
                --output-dir "$out_dir"
    else
        STEERING_POC_QWEN3_VECTOR="$(pwd)/$VECTOR" \
        STEERING_POC_QWEN3_LAYER="$LAYER" \
        STEERING_POC_QWEN3_ALPHA="$alpha" \
        STEERING_POC_QWEN3_NORMALIZE=1 \
        PYTHONUNBUFFERED=1 \
        python -m "$EXPORT_MODULE" \
            --device "$DEVICE" \
            --runtime "$RUNTIME" \
            --sequence-lengths "$SEQUENCE_LENGTHS" \
            --context-lengths "$CONTEXT_LENGTHS" \
            --skip-profiling \
            --output-dir "$out_dir"
    fi
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        printf "success\n" > "$status"
        log "Export succeeded label=$label"
    else
        printf "failed rc=%s\n" "$rc" > "$status"
        log "Export FAILED label=$label rc=$rc"
        exit "$rc"
    fi
}

log "Run dir: $RUN_DIR"
log "Log file: $LOG"
log "Phase: $PHASE"

case "$PHASE" in
    vector)
        apply_patch_if_needed
        ensure_vector
        printf "success\n" > "$RUN_DIR/vector.status"
        ;;
    baseline)
        apply_patch_if_needed
        ensure_vector
        run_export baseline
        ;;
    alpha0)
        apply_patch_if_needed
        ensure_vector
        run_export alpha_0 0
        ;;
    alpha4)
        apply_patch_if_needed
        ensure_vector
        run_export alpha_4 4
        ;;
    alphaneg4|alpha-4)
        apply_patch_if_needed
        ensure_vector
        run_export alpha_neg4 -4
        ;;
    all)
        apply_patch_if_needed
        ensure_vector
        run_export baseline
        run_export alpha_0 0
        run_export alpha_4 4
        run_export alpha_neg4 -4
        ;;
    *)
        log "Unknown phase '$PHASE'. Use: vector, baseline, alpha0, alpha4, alphaneg4, all"
        exit 2
        ;;
esac

log "Done phase=$PHASE"
