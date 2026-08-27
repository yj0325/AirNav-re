#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRNAV_ROOT:-$PWD}"
MODEL_REPO="${MODEL_REPO:-ddbcdd/airnav}"
DATASET_REPO="${DATASET_REPO:-ddbcdd/airnav}"
MAX_WORKERS="${MAX_WORKERS:-4}"

# The requested migration set. Optional unrelated AirNav weights are disabled.
MODEL_NAME="${MODEL_NAME:-AirNavMemoryColdStart}"
UPLOAD_BASE_MODEL="${UPLOAD_BASE_MODEL:-1}"
UPLOAD_DATA="${UPLOAD_DATA:-1}"
UPLOAD_ENV="${UPLOAD_ENV:-1}"
UPLOAD_LATEST_CHECKPOINT="${UPLOAD_LATEST_CHECKPOINT:-1}"

RUN_NAME="${RUN_NAME:-airnav_memory_grpo_coldstart_8gpu_b4_g4_u85}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/verl/checkpoints/airnav_memory_grpo/$RUN_NAME}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-113}"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/global_step_$CHECKPOINT_STEP"
PHOTO_ARCHIVE_DIR="$ROOT/archives/TrainPhotoData"
LARGE_DATA_ARCHIVE_DIR="$ROOT/archives/LargeData"

require_path() {
    if [[ ! -e "$1" ]]; then
        echo "Required path is missing: $1" >&2
        exit 2
    fi
}

if ! command -v modelscope >/dev/null 2>&1; then
    echo "modelscope CLI is not available in the active environment." >&2
    exit 2
fi

if [[ "$UPLOAD_BASE_MODEL" == "1" ]]; then
    require_path "$ROOT/model_weight/$MODEL_NAME"
    modelscope upload "$MODEL_REPO" \
        "$ROOT/model_weight/$MODEL_NAME" "$MODEL_NAME" \
        --repo-type model --max-workers "$MAX_WORKERS" \
        --commit-message "Upload AirNavMemoryColdStart model"
fi

if [[ "$UPLOAD_DATA" == "1" ]]; then
    require_path "$ROOT/data"
    require_path "$PHOTO_ARCHIVE_DIR/SHA256SUMS"
    require_path "$LARGE_DATA_ARCHIVE_DIR/SHA256SUMS"
    require_path "$LARGE_DATA_ARCHIVE_DIR/ORIGINAL_SHA256SUM"
    modelscope upload "$DATASET_REPO" \
        "$ROOT/data" data \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --exclude 'gsam/full_scan_*' 'data/gsam/full_scan_*' \
        --commit-message "Upload AirNav datasets"

    modelscope upload "$DATASET_REPO" \
        "$LARGE_DATA_ARCHIVE_DIR" archives/LargeData \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload sharded large AirNav data file"

    modelscope upload "$DATASET_REPO" \
        "$PHOTO_ARCHIVE_DIR" archives/TrainPhotoData \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload sharded GRPO training image archive"
fi

if [[ "$UPLOAD_LATEST_CHECKPOINT" == "1" ]]; then
    require_path "$CHECKPOINT_DIR"
    require_path "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
    if [[ "$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" != "$CHECKPOINT_STEP" ]]; then
        echo "Checkpoint tracker does not point to step $CHECKPOINT_STEP." >&2
        exit 3
    fi
    modelscope upload "$DATASET_REPO" \
        "$CHECKPOINT_DIR" \
        "checkpoints/airnav_memory_grpo/$RUN_NAME/global_step_$CHECKPOINT_STEP" \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload resumable official GRPO step $CHECKPOINT_STEP"

    modelscope upload "$DATASET_REPO" \
        "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" \
        "checkpoints/airnav_memory_grpo/$RUN_NAME/latest_checkpointed_iteration.txt" \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload official GRPO checkpoint tracker"
fi

if [[ "$UPLOAD_ENV" == "1" ]]; then
    require_path "$ROOT/environment/airnav-conda-pack.tar.gz"
    require_path "$ROOT/environment/SHA256SUMS"
    modelscope upload "$DATASET_REPO" \
        "$ROOT/environment/airnav-conda-pack.tar.gz" \
        environment/airnav-conda-pack.tar.gz \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload packed AirNav environment"
    modelscope upload "$DATASET_REPO" \
        "$ROOT/environment/SHA256SUMS" environment/SHA256SUMS \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload environment checksum"
fi

echo "ModelScope upload set completed."
