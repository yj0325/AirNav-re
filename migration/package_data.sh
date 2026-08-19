#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRNAV_ROOT:-/data1/jingyang/AirNav}"
TRANSFER_DIR="${TRANSFER_DIR:-/data1/jingyang/tmp/airnav_migration}"
PHOTO_SOURCE="$ROOT/official_grpo_repro/TrainPhotoData"
PHOTO_ARCHIVE_DIR="$TRANSFER_DIR/train_photo_data"
PART_SIZE="${PART_SIZE:-4G}"
PART_PREFIX="TrainPhotoData.tar.part-"

test -d "$PHOTO_SOURCE"
mkdir -p "$PHOTO_ARCHIVE_DIR"

if compgen -G "$PHOTO_ARCHIVE_DIR/$PART_PREFIX*" >/dev/null; then
    echo "Archive parts already exist; refusing to overwrite: $PHOTO_ARCHIVE_DIR" >&2
    exit 2
fi

tar -cf - -C "$ROOT/official_grpo_repro" TrainPhotoData \
    | split -b "$PART_SIZE" - "$PHOTO_ARCHIVE_DIR/$PART_PREFIX"

(
    cd "$PHOTO_ARCHIVE_DIR"
    sha256sum "$PART_PREFIX"* > SHA256SUMS
)

echo "TrainPhotoData archive parts:"
ls -lh "$PHOTO_ARCHIVE_DIR/$PART_PREFIX"* "$PHOTO_ARCHIVE_DIR/SHA256SUMS"
