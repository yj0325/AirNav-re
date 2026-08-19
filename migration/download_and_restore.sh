#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRNAV_ROOT:-$PWD}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_ROOT="${AIRNAV_ENV_ROOT:-$CONDA_ROOT/envs/airnav}"
HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
TMP_ROOT="${AIRNAV_TMP_ROOT:-/tmp/airnav-${USER:-user}}"
MODEL_REPO="${MODEL_REPO:-ddbcdd/airnav}"
DATASET_REPO="${DATASET_REPO:-ddbcdd/airnav}"
MAX_WORKERS="${MAX_WORKERS:-8}"
RUN_NAME="airnav_grpo_7gpu_resume17_optimized"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-150}"

if ! command -v modelscope >/dev/null 2>&1; then
    echo "Activate a small transfer environment containing modelscope first." >&2
    exit 2
fi
if [[ ! -d "$ROOT/.git" ]]; then
    echo "Clone https://gh-proxy.com/https://github.com/yj0325/AirNav to $ROOT first." >&2
    exit 3
fi
if [[ -d "$ENV_ROOT" ]] && [[ -n "$(find "$ENV_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to extract over non-empty environment: $ENV_ROOT" >&2
    exit 4
fi

mkdir -p "$ROOT/model_weight" "$HF_ROOT" "$TMP_ROOT/ray" "$ENV_ROOT"

# Repository paths were uploaded relative to ROOT, so downloading at ROOT
# restores data, TrainPhotoData, the checkpoint, tracker, and environment.
modelscope download "$DATASET_REPO" --repo-type dataset \
    --local_dir "$ROOT" --max-workers "$MAX_WORKERS"
modelscope download "$MODEL_REPO" --repo-type model \
    --local_dir "$ROOT/model_weight" --max-workers "$MAX_WORKERS"

CHECKPOINT_ROOT="$ROOT/official_grpo_repro/checkpoints/$RUN_NAME"
test -d "$ROOT/model_weight/AirNavSFT"
test -d "$ROOT/data/AirNav_GRPO"
test -d "$ROOT/official_grpo_repro/TrainPhotoData"
test -d "$CHECKPOINT_ROOT/global_step_$CHECKPOINT_STEP"
test "$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" = "$CHECKPOINT_STEP"

cd "$ROOT/environment"
sha256sum -c SHA256SUMS
tar -xzf airnav-conda-pack.tar.gz -C "$ENV_ROOT"
"$ENV_ROOT/bin/conda-unpack"

"$ENV_ROOT/bin/python" "$ROOT/migration/relocate_airnav.py" \
    --project-root "$ROOT" \
    --new-root "$ROOT" \
    --new-conda "$CONDA_ROOT" \
    --new-hf "$HF_ROOT" \
    --new-tmp "$TMP_ROOT"

"$ENV_ROOT/bin/python" -m pip install --no-deps -e "$ROOT/verl"

if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    "$CONDA_ROOT/bin/conda" env config vars set -p "$ENV_ROOT" \
        PYTHONNOUSERSITE=1 \
        PYTHONPATH="$ROOT:$ROOT/verl" \
        AIRNAV_REPRO_ROOT="$ROOT/official_grpo_repro" \
        AIRNAV_CONDA_ROOT="$CONDA_ROOT" \
        AIRNAV_ENV_ROOT="$ENV_ROOT" \
        AIRNAV_TMP_ROOT="$TMP_ROOT" \
        HF_ENDPOINT=https://hf-mirror.com \
        HF_HOME="$HF_ROOT" \
        HF_HUB_CACHE="$HF_ROOT/hub" \
        HF_DATASETS_CACHE="$HF_ROOT/datasets" \
        HF_ASSETS_CACHE="$HF_ROOT/assets" \
        TMPDIR="$TMP_ROOT" \
        RAY_TMPDIR="$TMP_ROOT/ray" \
        MPLCONFIGDIR="$TMP_ROOT/matplotlib"
else
    echo "Warning: $CONDA_ROOT/bin/conda is absent; environment variables were not persisted." >&2
fi

"$ENV_ROOT/bin/python" - <<'PY'
import sys
import torch
import transformers
import vllm
import ray
import verl

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("ray:", ray.__version__)
print("verl:", verl.__file__)
PY

echo "Restore complete. Activate with:"
echo "source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $ENV_ROOT"
echo "Resume checkpoint: $CHECKPOINT_ROOT/global_step_$CHECKPOINT_STEP"
