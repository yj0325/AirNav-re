#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIRNAV_ROOT:-$PWD}"
TRANSFER_DIR="${TRANSFER_DIR:-$ROOT/archives}"
PHOTO_SOURCE="${PHOTO_SOURCE:-$ROOT/official_grpo_repro/TrainPhotoData}"
PHOTO_ARCHIVE_DIR="$TRANSFER_DIR/TrainPhotoData"
LARGE_DATA_SOURCE="$ROOT/data/gsam/full_scan_(100, 240, 410).npz"
LARGE_DATA_ARCHIVE_DIR="$TRANSFER_DIR/LargeData"
PART_SIZE="${PART_SIZE:-4G}"
PART_PREFIX="TrainPhotoData.tar.part-"
LARGE_PART_SIZE="${LARGE_PART_SIZE:-1G}"
LARGE_PART_PREFIX="full_scan.npz.part-"

test -d "$PHOTO_SOURCE"
mkdir -p "$PHOTO_ARCHIVE_DIR"

if compgen -G "$PHOTO_ARCHIVE_DIR/$PART_PREFIX*" >/dev/null; then
    test -f "$PHOTO_ARCHIVE_DIR/SHA256SUMS"
    echo "Reusing existing TrainPhotoData parts."
else
    tar -cf - -C "$ROOT/official_grpo_repro" TrainPhotoData \
        | split -b "$PART_SIZE" - "$PHOTO_ARCHIVE_DIR/$PART_PREFIX"
    (
        cd "$PHOTO_ARCHIVE_DIR"
        sha256sum "$PART_PREFIX"* > SHA256SUMS
    )
fi

echo "TrainPhotoData archive parts:"
ls -lh "$PHOTO_ARCHIVE_DIR/$PART_PREFIX"* "$PHOTO_ARCHIVE_DIR/SHA256SUMS"

test -f "$LARGE_DATA_SOURCE"
mkdir -p "$LARGE_DATA_ARCHIVE_DIR"
if compgen -G "$LARGE_DATA_ARCHIVE_DIR/$LARGE_PART_PREFIX*" >/dev/null; then
    test -f "$LARGE_DATA_ARCHIVE_DIR/SHA256SUMS"
    test -f "$LARGE_DATA_ARCHIVE_DIR/ORIGINAL_SHA256SUM"
    echo "Reusing existing large-data parts."
else
    split -b "$LARGE_PART_SIZE" "$LARGE_DATA_SOURCE" \
        "$LARGE_DATA_ARCHIVE_DIR/$LARGE_PART_PREFIX"
    (
        cd "$LARGE_DATA_ARCHIVE_DIR"
        sha256sum "$LARGE_PART_PREFIX"* > SHA256SUMS
    )
    (
        cd "$(dirname "$LARGE_DATA_SOURCE")"
        sha256sum "$(basename "$LARGE_DATA_SOURCE")" \
            > "$LARGE_DATA_ARCHIVE_DIR/ORIGINAL_SHA256SUM"
    )
fi

echo "Large-data parts:"
ls -lh "$LARGE_DATA_ARCHIVE_DIR"
