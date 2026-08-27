#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/nfsdata/yangjing/AirNav}
PYTHON_BIN=${PYTHON_BIN:-/nfsdata/yangjing/miniconda/envs/airnav/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/data/airnav_memory}

export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT_DIR}/verl:${ROOT_DIR}:${PYTHONPATH:-}

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/verl/examples/data_preprocess/AirNav_episode.py" \
  --info "${ROOT_DIR}/data/AirNav/train/info_train.json" \
  --airnav "${ROOT_DIR}/data/AirNav/train/airnav_train.json" \
  --split train \
  --output "${OUTPUT_DIR}/train.parquet"

"${PYTHON_BIN}" "${ROOT_DIR}/verl/examples/data_preprocess/AirNav_episode.py" \
  --info "${ROOT_DIR}/data/AirNav/val/info_val_seen.json" \
  --airnav "${ROOT_DIR}/data/AirNav/val/airnav_val_seen.json" \
  --split val_seen \
  --output "${OUTPUT_DIR}/val_seen.parquet"

"${PYTHON_BIN}" "${ROOT_DIR}/verl/examples/data_preprocess/AirNav_episode.py" \
  --info "${ROOT_DIR}/data/AirNav/val/info_val_unseen.json" \
  --airnav "${ROOT_DIR}/data/AirNav/val/airnav_val_unseen.json" \
  --split val_unseen \
  --output "${OUTPUT_DIR}/val_unseen.parquet"
