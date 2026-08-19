#!/usr/bin/env bash
# Backward-compatible entry point; the restore procedure is server-independent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/download_and_restore.sh" "$@"
