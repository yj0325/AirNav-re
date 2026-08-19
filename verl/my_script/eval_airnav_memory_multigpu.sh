#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/data1/jingyang/AirNav}
PYTHON_BIN=${PYTHON_BIN:-/data1/jingyang/miniconda3/envs/airnav/bin/python}
MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/verl/checkpoints/airnav_memory_grpo/airnav_sft_memory_terminal_fullbatch_g5_b14_20260814/global_step_200/actor/huggingface}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-airnav-memory-step200}
POLICY=${POLICY:-learned}
GPU_IDS=${GPU_IDS:-0,2,3,4,5,6,7}
PORT_BASE=${PORT_BASE:-8100}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-8}
ONE_INSTRUCTION_PER_EPISODE=${ONE_INSTRUCTION_PER_EPISODE:-1}
PROGRESS_LOG_INTERVAL=${PROGRESS_LOG_INTERVAL:-50}
OUTPUT=${OUTPUT:-${ROOT_DIR}/result/memory_global_step_200_one_instruction_per_episode_eval.json}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-${ROOT_DIR}/logs/eval_memory_global_step_200_one_instruction_servers}

export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT_DIR}:${ROOT_DIR}/verl:${PYTHONPATH:-}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/data1/jingyang/tmp/matplotlib_airnav_eval}

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
mkdir -p "${SERVER_LOG_DIR}" "$(dirname "${OUTPUT}")" "${MPLCONFIGDIR}"

pids=()
base_url_args=()
evaluation_args=()
if [[ "${ONE_INSTRUCTION_PER_EPISODE}" == "1" ]]; then
  evaluation_args+=(--one-instruction-per-episode)
fi
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  port=$((PORT_BASE + index))
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --limit-mm-per-prompt '{"image":5,"video":0}' \
    --max-model-len 4352 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 32768 \
    --disable-log-requests \
    --disable-uvicorn-access-log \
    --port "${port}" \
    >"${SERVER_LOG_DIR}/gpu_${gpu}_port_${port}.log" 2>&1 &
  pids+=("$!")
  base_url_args+=(--base-url "http://127.0.0.1:${port}/v1")
done

for index in "${!GPUS[@]}"; do
  port=$((PORT_BASE + index))
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "${ready}" != 1 ]]; then
    echo "vLLM server on port ${port} failed to become ready" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" "${ROOT_DIR}/airnav_memory_eval.py" \
  --model-name "${SERVED_MODEL_NAME}" \
  --policy "${POLICY}" \
  --temperature 1.0 \
  --workers "$((WORKERS_PER_GPU * ${#GPUS[@]}))" \
  --progress-log-interval "${PROGRESS_LOG_INTERVAL}" \
  --fail-on-api-error \
  --output "${OUTPUT}" \
  "${base_url_args[@]}" \
  "${evaluation_args[@]}" \
  "$@"
