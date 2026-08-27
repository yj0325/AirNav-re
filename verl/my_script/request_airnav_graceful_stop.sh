#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/nfsdata/yangjing/AirNav}
EXPERIMENT_NAME=${1:?usage: request_airnav_graceful_stop.sh EXPERIMENT_NAME}
STOP_FILE=${ROOT_DIR}/verl/checkpoints/airnav_memory_grpo/${EXPERIMENT_NAME}/STOP_AFTER_CHECKPOINT

mkdir -p "$(dirname "${STOP_FILE}")"
touch "${STOP_FILE}"
echo "Graceful stop requested: ${STOP_FILE}"
echo "Training will finish the current global step, save a checkpoint, and then exit."
