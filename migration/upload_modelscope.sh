#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRNAV_ROOT:-/data1/jingyang/AirNav}"
MODEL_REPO="${MODEL_REPO:-ddbcdd/airnav}"
DATASET_REPO="${DATASET_REPO:-ddbcdd/airnav}"
TRANSFER_DIR="${TRANSFER_DIR:-/data1/jingyang/tmp/airnav_migration}"
PHOTO_ARCHIVE_DIR="$TRANSFER_DIR/train_photo_data"
LARGE_DATA_ARCHIVE_DIR="$TRANSFER_DIR/large_data_file"
MAX_WORKERS="${MAX_WORKERS:-4}"

# The requested migration set. Optional unrelated AirNav weights are disabled.
UPLOAD_BASE_MODEL="${UPLOAD_BASE_MODEL:-1}"
UPLOAD_DATA="${UPLOAD_DATA:-1}"
UPLOAD_ENV="${UPLOAD_ENV:-1}"
UPLOAD_LATEST_CHECKPOINT="${UPLOAD_LATEST_CHECKPOINT:-1}"

RUN_NAME="airnav_grpo_7gpu_resume17_optimized"
CHECKPOINT_ROOT="$ROOT/official_grpo_repro/checkpoints/$RUN_NAME"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-150}"
CHECKPOINT_DIR="$CHECKPOINT_ROOT/global_step_$CHECKPOINT_STEP"

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
    require_path "$ROOT/model_weight/AirNavSFT"
    modelscope upload "$MODEL_REPO" \
        "$ROOT/model_weight/AirNavSFT" AirNavSFT \
        --repo-type model --max-workers "$MAX_WORKERS" \
        --commit-message "Upload AirNavSFT base model"
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
        "official_grpo_repro/checkpoints/$RUN_NAME/global_step_$CHECKPOINT_STEP" \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload resumable official GRPO step $CHECKPOINT_STEP"

    modelscope upload "$DATASET_REPO" \
        "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" \
        "official_grpo_repro/checkpoints/$RUN_NAME/latest_checkpointed_iteration.txt" \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload official GRPO checkpoint tracker"
fi

if [[ "$UPLOAD_ENV" == "1" ]]; then
    require_path "$TRANSFER_DIR/airnav-conda-pack.tar.gz"
    require_path "$TRANSFER_DIR/SHA256SUMS"
    modelscope upload "$DATASET_REPO" \
        "$TRANSFER_DIR/airnav-conda-pack.tar.gz" \
        environment/airnav-conda-pack.tar.gz \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload packed AirNav environment"
    modelscope upload "$DATASET_REPO" \
        "$TRANSFER_DIR/SHA256SUMS" environment/SHA256SUMS \
        --repo-type dataset --max-workers "$MAX_WORKERS" \
        --commit-message "Upload environment checksum"
fi

echo "ModelScope upload set completed."
