#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR=${ROOT_DIR:-/nfsdata/yangjing/AirNav}
PYTHON_BIN=${PYTHON_BIN:-/nfsdata/yangjing/miniconda/envs/airnav/bin/python}
ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then
  shift
fi
N_GPUS=${N_GPUS:-7}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-14}
ROLLOUT_GROUP_SIZE=${ROLLOUT_GROUP_SIZE:-5}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-14}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
ACTOR_DYNAMIC_BSZ=${ACTOR_DYNAMIC_BSZ:-true}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-8192}
LOG_PROB_DYNAMIC_BSZ=${LOG_PROB_DYNAMIC_BSZ:-true}
OLD_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${OLD_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}
TRAIN_FILE=${TRAIN_FILE:-${ROOT_DIR}/data/airnav_memory/train_one_instruction_seed1.parquet}
MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/model_weight/AirNavSFT}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-airnav_sft_online_memory_terminal_fullbatch_g5_7gpu_b14}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1500}
SAVE_FREQ=${SAVE_FREQ:-50}
RESUME_MODE=${RESUME_MODE:-auto}
CHECKPOINT_CONTENTS=${CHECKPOINT_CONTENTS:-'["model","optimizer","extra","hf_model"]'}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
STOP_FILE=${STOP_FILE:-${ROOT_DIR}/verl/checkpoints/airnav_memory_grpo/${EXPERIMENT_NAME}/STOP_AFTER_CHECKPOINT}

export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT_DIR}/verl:${ROOT_DIR}:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-offline}
export USE_OPTIMIZED_MODEL=0
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VERL_AUTO_PADDING=${VERL_AUTO_PADDING:-1}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/nfsdata/yangjing/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/nfsdata/yangjing/.cache/huggingface/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/nfsdata/yangjing/.cache/huggingface/datasets}
export HF_ASSETS_CACHE=${HF_ASSETS_CACHE:-/nfsdata/yangjing/.cache/huggingface/assets}
export TMPDIR=${TMPDIR:-/nfsdata/yangjing/.cache/huggingface/tmp}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/airnav-yangjing/matplotlib_airnav}

mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "$(dirname "${STOP_FILE}")"
rm -f "${STOP_FILE}"

cd "${ROOT_DIR}/verl"

"${PYTHON_BIN}" -m verl.trainer.benchmark_trainer \
  algorithm.adv_estimator=airnav_episode_grpo \
  +algorithm.airnav_return_beta=0.95 \
  +algorithm.airnav_adv_epsilon=1e-6 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${ROOT_DIR}/data/airnav_memory/val_seen.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.return_raw_chat=true \
  data.return_multi_modal_inputs=false \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${ACTOR_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_CONTENTS}" \
  actor_rollout_ref.actor.checkpoint.load_contents="${CHECKPOINT_CONTENTS}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_GROUP_SIZE}" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${LOG_PROB_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${OLD_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=airnav_memory \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${ROOT_DIR}/verl/my_script/airnav_memory_agent.yaml" \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:5,video:0}' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${LOG_PROB_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  algorithm.use_kl_in_reward=false \
  reward_model.enable=false \
  custom_reward_function.path=null \
  critic.enable=false \
  trainer.balance_batch=true \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=airnav_memory_grpo \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  +trainer.stop_file="${STOP_FILE}" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  "$@"
