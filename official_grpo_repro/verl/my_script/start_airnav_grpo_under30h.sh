#!/usr/bin/env bash
set -euo pipefail

ROOT="/nfsdata/yangjing/AirNav/official_grpo_repro"
LOG="$ROOT/logs/airnav_grpo_7gpu_optimized.log"
PID_FILE="$ROOT/run/airnav_grpo_7gpu_optimized.pid"

mkdir -p "$ROOT/logs" "$ROOT/run"
: > "$LOG"

nohup bash "$ROOT/verl/my_script/run_airnav_grpo_7gpu.sh" >> "$LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
printf 'AirNav GRPO nohup PID: %s\nLog: %s\n' "$TRAIN_PID" "$LOG"

# Keep the launcher session alive so managed execution does not reap the
# detached process. nohup still protects the training process from SIGHUP.
wait "$TRAIN_PID"
