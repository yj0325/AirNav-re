#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AIRNAV_REPRO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERL_ROOT="$ROOT/verl"
RUN_ROOT="$ROOT/run"
LOG_ROOT="$ROOT/logs"
CHECKPOINT_ROOT="$ROOT/checkpoints/airnav_grpo_7gpu_resume17_optimized"
STOP_FILE="$RUN_ROOT/airnav_grpo.stop"
CONDA_ROOT="${AIRNAV_CONDA_ROOT:-$HOME/miniconda3}"
ENV_ROOT="${AIRNAV_ENV_ROOT:-$CONDA_ROOT/envs/airnav}"
LATEST_FILE="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"

if [[ -n "${AIRNAV_RESUME_CHECKPOINT:-}" ]]; then
    SOURCE_CHECKPOINT="$AIRNAV_RESUME_CHECKPOINT"
elif [[ -f "$LATEST_FILE" ]]; then
    LATEST_STEP="$(tr -d '[:space:]' < "$LATEST_FILE")"
    SOURCE_CHECKPOINT="$CHECKPOINT_ROOT/global_step_$LATEST_STEP"
else
    echo "Missing checkpoint tracker: $LATEST_FILE" >&2
    exit 3
fi

AIRNAV_TMP_ROOT="${AIRNAV_TMP_ROOT:-${TMPDIR:-/tmp/airnav-${USER:-user}}}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$AIRNAV_TMP_ROOT/ray"
rm -f "$STOP_FILE"

if [[ ! -d "$SOURCE_CHECKPOINT/actor" ]]; then
    echo "Missing resume checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 3
fi

# Seven-card run.  GPU 1 is excluded because it is occupied by another job;
# override this variable only when a different seven-card allocation is
# explicitly coordinated.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7}"
export PYTHONPATH="$VERL_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export USE_OPTIMIZED_MODEL=0
export TOKENIZERS_PARALLELISM=true
export TMPDIR="$AIRNAV_TMP_ROOT"
export RAY_TMPDIR="$AIRNAV_TMP_ROOT/ray"
export HF_HOME="${HF_HOME:-$AIRNAV_TMP_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$AIRNAV_TMP_ROOT/matplotlib}"
# Parameter placement is an engineering-only switch: reference weights are
# frozen and the objective is unchanged. Set AIRNAV_REF_PARAM_OFFLOAD=true to
# restore the original CPU-offload behavior if resident-GPU mode OOMs.
REF_PARAM_OFFLOAD="${AIRNAV_REF_PARAM_OFFLOAD:-false}"

shopt -s nullglob
TRAIN_FILES=("$ROOT"/data/AirNav_GRPO/train/part-*.parquet)
VAL_FILES=("$ROOT"/data/AirNav_GRPO/val_seen/part-*.parquet)
if (( ${#TRAIN_FILES[@]} == 0 || ${#VAL_FILES[@]} == 0 )); then
    echo "GRPO parquet data is incomplete; run scripts/prepare_airnav_grpo.py first." >&2
    exit 2
fi

cd "$VERL_ROOT"
exec "$CONDA_ROOT/bin/conda" run --no-capture-output -p "$ENV_ROOT" \
    python -u -m verl.trainer.benchmark_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.reuse_rollout_log_probs_as_old=true \
    data.train_files="$ROOT/data/AirNav_GRPO/train/part-*.parquet" \
    data.val_files="$ROOT/data/AirNav_GRPO/val_seen/part-*.parquet" \
    data.train_batch_size=98 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=false \
    data.filter_overlong_prompts_workers=128 \
    data.truncation=error \
    data.image_key=images \
    +data.cache_dir="$AIRNAV_TMP_ROOT/dataset_cache" \
    actor_rollout_ref.model.path="$ROOT/model_weight/AirNavSFT" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=98 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=14 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.freeze_vision_tower=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=14 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=14 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    custom_reward_function.path="$VERL_ROOT/reward_fn/AirNav_rl.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=AirNav_GRPO_reproduction \
    trainer.experiment_name=AirNav_rl_7gpu_batch98_group5_step1500_resume17_optimized \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.total_training_steps=1500 \
    trainer.total_epochs=30 \
    trainer.default_local_dir="$CHECKPOINT_ROOT" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$SOURCE_CHECKPOINT" \
    +trainer.stop_file="$STOP_FILE" \
    "$@"
