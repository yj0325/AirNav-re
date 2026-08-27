#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${AIRNAV_ROOT:-$PWD}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda}"
ENV_ROOT="${AIRNAV_ENV_ROOT:-$CONDA_ROOT/envs/airnav}"
HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
TMP_ROOT="${AIRNAV_TMP_ROOT:-/tmp/airnav-${USER:-user}}"
ARCHIVE="$ROOT/environment/airnav-conda-pack.tar.gz"
STATE_ROOT="${AIRNAV_RESUME_STATE_ROOT:-$ROOT/.airnav-resume}"
RUN_NAME="${RUN_NAME:-airnav_memory_grpo_coldstart_8gpu_b4_g4_u85}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-150}"
CHECKPOINT_ROOT="$ROOT/verl/checkpoints/airnav_memory_grpo/$RUN_NAME"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

on_error() {
    local exit_code=$?
    log "ERROR: resume failed at line $1 (exit $exit_code)"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

test -d "$ROOT/.git"
test -f "$ARCHIVE"
test -d "$ENV_ROOT"
MODEL_NAME="${MODEL_NAME:-AirNavMemoryColdStart}"
test -d "$ROOT/model_weight/$MODEL_NAME"
test -d "$ROOT/data/AirNav_GRPO"
test -d "$ROOT/official_grpo_repro/TrainPhotoData"
test -d "$CHECKPOINT_ROOT/global_step_$CHECKPOINT_STEP"
test "$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" = "$CHECKPOINT_STEP"

env_owner_uid="$(stat -c '%u' "$ENV_ROOT")"
if [[ "$(id -u)" != "$env_owner_uid" ]]; then
    log "ERROR: run as the owner of $ENV_ROOT (uid $env_owner_uid), current uid is $(id -u)"
    exit 2
fi

mkdir -p "$TMP_ROOT/ray" "$TMP_ROOT/matplotlib" "$STATE_ROOT"
exec 9>"$TMP_ROOT/resume_environment_restore.lock"
if ! flock -n 9; then
    log "ERROR: another environment restore is already running"
    exit 3
fi

if [[ "${AIRNAV_RECHECK_ARCHIVE:-0}" == "1" ]]; then
    log "Rechecking environment archive SHA256"
    (
        cd "$ROOT/environment"
        sha256sum -c SHA256SUMS
    )
else
    log "Skipping repeat SHA256 scan; the original restore already passed it before extraction"
fi

# A regular file left open when tar receives SIGHUP may be only partially
# written. GNU tar applies the archived mtime only after completing a file, so
# files newer than the archive are conservatively quarantined and re-extracted.
quarantine="$STATE_ROOT/recovery_env_partial_$(date '+%Y%m%d_%H%M%S')"
partial_count=0
while IFS= read -r -d '' partial_file; do
    relative_path="${partial_file#"$ENV_ROOT"/}"
    mkdir -p "$quarantine/$(dirname "$relative_path")"
    mv "$partial_file" "$quarantine/$relative_path"
    partial_count=$((partial_count + 1))
done < <(find "$ENV_ROOT" -xdev -type f -newer "$ARCHIVE" -print0)

if (( partial_count == 0 )); then
    rmdir "$quarantine"
    log "No potentially partial regular files found"
else
    log "Quarantined $partial_count potentially partial files at $quarantine"
fi

log "Resuming archive extraction; existing complete files will be skipped"
tar --skip-old-files -xzf "$ARCHIVE" -C "$ENV_ROOT"

test -x "$ENV_ROOT/bin/python"
test -x "$ENV_ROOT/bin/conda-unpack"

log "Running conda-unpack"
"$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"

log "Relocating project and cache paths"
"$ENV_ROOT/bin/python" "$ROOT/migration/relocate_airnav.py" \
    --project-root "$ROOT" \
    --new-root "$ROOT" \
    --new-conda "$CONDA_ROOT" \
    --new-hf "$HF_ROOT" \
    --new-tmp "$TMP_ROOT"

log "Installing the restored VERL checkout"
"$ENV_ROOT/bin/python" -m pip install --no-deps -e "$ROOT/verl"

if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
    log "Persisting environment variables"
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
    log "WARNING: $CONDA_ROOT/bin/conda is absent; environment variables were not persisted"
fi

log "Verifying restored Python environment"
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

date '+%F %T' > "$STATE_ROOT/resume_environment_restore.complete"
log "Restore complete"
