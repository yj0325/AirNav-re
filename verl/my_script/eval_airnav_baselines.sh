#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/nfsdata/yangjing/AirNav}
PYTHON_BIN=${PYTHON_BIN:-/nfsdata/yangjing/miniconda/envs/airnav/bin/python}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MODEL_KIND=${1:-sft}

case "${MODEL_KIND}" in
  sft)
    MODEL_PATH=${ROOT_DIR}/model_weight/AirNavSFT
    POLICY=fixed
    ;;
  r1)
    MODEL_PATH=${ROOT_DIR}/model_weight/AirVLN-R1
    POLICY=fixed
    ;;
  memory)
    MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a merged VERL actor checkpoint for memory evaluation}
    POLICY=learned
    ;;
  *)
    echo "usage: $0 {sft|r1|memory}" >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name airnav \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --max-model-len 4352 \
  --port 8000 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

until curl -fsS http://localhost:8000/v1/models >/dev/null; do
  sleep 2
done

"${PYTHON_BIN}" "${ROOT_DIR}/airnav_memory_eval.py" \
  --model-name airnav \
  --policy "${POLICY}" \
  --output "${ROOT_DIR}/result/${MODEL_KIND}_airnav_eval.json"
