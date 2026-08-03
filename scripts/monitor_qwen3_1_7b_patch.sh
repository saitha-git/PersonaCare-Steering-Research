#!/usr/bin/env bash
# Monitor the latest Qwen3-1.7B patch export run from another WSL terminal.
#
# Usage:
#   bash scripts/monitor_qwen3_1_7b_patch.sh
#   INTERVAL=15 TAIL_LINES=80 bash scripts/monitor_qwen3_1_7b_patch.sh
#   bash scripts/monitor_qwen3_1_7b_patch.sh artifacts/qwen3_1_7b_patch/runs/<run_id>
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_ROOT="${OUT_ROOT:-artifacts/qwen3_1_7b_patch}"
INTERVAL="${INTERVAL:-30}"
TAIL_LINES="${TAIL_LINES:-60}"
RUN_DIR="${1:-}"

if [ -z "$RUN_DIR" ]; then
    if [ ! -f "$OUT_ROOT/latest_run_dir.txt" ]; then
        echo "$OUT_ROOT/latest_run_dir.txt not found."
        echo "Start a run first, for example:"
        echo "  bash scripts/run_qwen3_1_7b_patch_step.sh baseline"
        exit 2
    fi
    RUN_DIR="$(cat "$OUT_ROOT/latest_run_dir.txt")"
fi

LOG="$RUN_DIR/run.log"

echo "Monitoring run dir: $RUN_DIR"
echo "Log file: $LOG"
echo "Refresh interval: ${INTERVAL}s"
echo

while true; do
    echo "================================================================"
    date '+%Y-%m-%d %H:%M:%S'
    echo

    echo "-- Runner status files --"
    if compgen -G "$RUN_DIR/*.status" >/dev/null; then
        for f in "$RUN_DIR"/*.status; do
            printf "%s: " "$(basename "$f")"
            cat "$f"
        done
    else
        echo "No phase status files yet."
    fi
    if [ -f "$RUN_DIR/runner.pid" ]; then
        printf "runner.pid: "
        cat "$RUN_DIR/runner.pid"
    fi
    echo

    echo "-- Matching processes --"
    ps -eo pid,ppid,etime,pcpu,pmem,rss,stat,cmd \
        | grep -E "run_qwen3_1_7b_patch_step|qai_hub_models.models.qwen3_1_7b.export|steering_poc.extract" \
        | grep -v grep || echo "No matching runner/export process found."
    echo

    echo "-- Output directories --"
    du -sh "$OUT_ROOT" 2>/dev/null || true
    find "$OUT_ROOT" -maxdepth 3 -type f \
        -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" 2>/dev/null \
        | sort \
        | tail -30 || true
    echo

    echo "-- Last ${TAIL_LINES} log lines --"
    if [ -f "$LOG" ]; then
        tail -n "$TAIL_LINES" "$LOG"
    else
        echo "Log file not written yet."
    fi
    echo

    sleep "$INTERVAL"
done
