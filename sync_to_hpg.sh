#!/usr/bin/env bash

set -euo pipefail

# Safe defaults:
# - Dry run unless --apply is provided.
# - Never sync local outputs/ to remote.
# - Never delete remote files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

LOCAL_DIR="${LOCAL_DIR:-${PROJECT_ROOT}/}"
REMOTE_USER="${REMOTE_USER:-vinhnguyen1}"
REMOTE_HOST="${REMOTE_HOST:-rsync.rc.ufl.edu}"
REMOTE_DIR="${REMOTE_DIR:-/blue/thai/vinhnguyen1/ia-unforeseen-70b/}"

MODE="dry-run"
EXTRA_RSYNC_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./sync_to_hpg.sh [--dry-run] [--apply] [--checksum] [--host HOST] [--user USER] [--remote-dir PATH] [--local-dir PATH] [--] [extra rsync args]

Examples:
  ./sync_to_hpg.sh
  ./sync_to_hpg.sh --apply
  ./sync_to_hpg.sh --apply --host hpg.rc.ufl.edu
  ./sync_to_hpg.sh --apply --checksum

Notes:
  - Default mode is --dry-run for safety.
  - outputs/ is excluded so remote experiment outputs are never overwritten by this script.
  - This script does not use --delete.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            MODE="dry-run"
            shift
            ;;
        --apply)
            MODE="apply"
            shift
            ;;
        --checksum)
            EXTRA_RSYNC_ARGS+=("--checksum")
            shift
            ;;
        --host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --remote-dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_RSYNC_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "$LOCAL_DIR" != */ ]]; then
    LOCAL_DIR="${LOCAL_DIR}/"
fi

if [[ "$REMOTE_DIR" != */ ]]; then
    REMOTE_DIR="${REMOTE_DIR}/"
fi

RSYNC_ARGS=(
    -avh
    --progress
    --itemize-changes
    --exclude=.git/
    --exclude=.venv/
    --exclude=.pytest_cache/
    --exclude=__pycache__/
    --exclude='*.pyc'
    --exclude=outputs/
    --exclude=checkpoints/
    --exclude=runtime_adapters/
    --exclude=models/
    --exclude=datasets/
    --exclude=logs/
    --exclude=wandb/
    --include=/repos/
    --include=/repos/introspection-adapters/
    --include=/repos/introspection-adapters/src/
    --include=/repos/introspection-adapters/src/finetuning/
    --include=/repos/introspection-adapters/src/finetuning/metalora.py
    --exclude=/repos/***
    --exclude='slurm-*.out'
)

if [[ "$MODE" == "dry-run" ]]; then
    RSYNC_ARGS+=(--dry-run)
fi

RSYNC_ARGS+=("${EXTRA_RSYNC_ARGS[@]}")

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "Sync mode: $MODE"
echo "Local dir:  $LOCAL_DIR"
echo "Remote dir: $REMOTE_TARGET"
echo

rsync "${RSYNC_ARGS[@]}" "$LOCAL_DIR" "$REMOTE_TARGET"

echo
echo "Done."
