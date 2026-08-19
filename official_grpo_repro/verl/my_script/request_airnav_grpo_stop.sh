#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/jingyang/AirNav/official_grpo_repro"
STOP_FILE="$ROOT/run/airnav_grpo.stop"
mkdir -p "$ROOT/run"
touch "$STOP_FILE"
echo "Graceful stop requested: $STOP_FILE"
echo "Training will finish the current global step, save a checkpoint, and exit."
