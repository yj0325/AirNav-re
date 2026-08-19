#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIRNAV_REPRO_ROOT:-$SCRIPT_DIR}"
CONDA_ROOT="${AIRNAV_CONDA_ROOT:-$HOME/miniconda3}"
ENV_ROOT="${AIRNAV_ENV_ROOT:-$CONDA_ROOT/envs/airnav}"
RUN_ROOT="$ROOT/run"
LOG_ROOT="$ROOT/logs"
PREPARE_LOG="$LOG_ROOT/airnav_grpo_prepare.log"
TRAIN_LOG="$LOG_ROOT/airnav_grpo_train.log"

AIRNAV_TMP_ROOT="${AIRNAV_TMP_ROOT:-${TMPDIR:-/tmp/airnav-${USER:-user}}}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$AIRNAV_TMP_ROOT"
echo $$ > "$RUN_ROOT/airnav_grpo_pipeline.pid"
trap 'rm -f "$RUN_ROOT/airnav_grpo_pipeline.pid"' EXIT

export TMPDIR="$AIRNAV_TMP_ROOT"
export HF_HOME="${HF_HOME:-$AIRNAV_TMP_ROOT/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$AIRNAV_TMP_ROOT/matplotlib}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date '+%F %T')] stage=preprocess start" | tee -a "$PREPARE_LOG"
cd "$ROOT"
"$CONDA_ROOT/bin/conda" run --no-capture-output -p "$ENV_ROOT" \
    python -u "$ROOT/scripts/prepare_airnav_grpo.py" --split all \
    2>&1 | tee -a "$PREPARE_LOG"
echo "[$(date '+%F %T')] stage=preprocess complete" | tee -a "$PREPARE_LOG"

echo "[$(date '+%F %T')] stage=train start gpus=0,2,3,4,5,6,7" | tee -a "$TRAIN_LOG"
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 \
    "$ROOT/verl/my_script/run_airnav_grpo_7gpu.sh" \
    2>&1 | tee -a "$TRAIN_LOG"
echo "[$(date '+%F %T')] stage=train complete" | tee -a "$TRAIN_LOG"
