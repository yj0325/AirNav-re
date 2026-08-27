#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENV="${SOURCE_ENV:-${CONDA_PREFIX:-$PWD/.venv}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/environment}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/migration/manifests"

if ! command -v conda-pack >/dev/null 2>&1; then
    echo "conda-pack is required in the current transfer environment." >&2
    exit 2
fi

conda-pack \
    -p "$SOURCE_ENV" \
    -o "$OUTPUT_DIR/airnav-conda-pack.tar.gz" \
    --ignore-editable-packages \
    --ignore-missing-files \
    --force

(
    cd "$OUTPUT_DIR"
    sha256sum airnav-conda-pack.tar.gz > SHA256SUMS
)

"$SOURCE_ENV/bin/python" -m pip freeze --all \
    > "$PROJECT_ROOT/migration/manifests/pip-freeze-airnav.txt"

"${CONDA_EXE:-conda}" list \
    -p "$SOURCE_ENV" --explicit \
    > "$PROJECT_ROOT/migration/manifests/conda-linux-64-explicit.txt"

"${CONDA_EXE:-conda}" env export \
    -p "$SOURCE_ENV" --no-builds \
    | sed '/^prefix:/d' \
    > "$PROJECT_ROOT/migration/manifests/environment-airnav.yml"

echo "Environment package: $OUTPUT_DIR/airnav-conda-pack.tar.gz"
cat "$OUTPUT_DIR/SHA256SUMS"
