#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:-$PROJECT/envs/audit_env/bin/python3}"
: "${AUDIT_CONFIG:?Set AUDIT_CONFIG to a persistent resolved or frozen config path}"

export PROJECT PYTHON
export HF_HOME="${HF_HOME:-${SLURM_TMPDIR:-/tmp}/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
  module load cuda/12.8.1
  export LD_LIBRARY_PATH="${CUDA_HOME:-/opt/nvidia/hpc_sdk}/Linux_x86_64/12.8/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$AUDIT_CONFIG" ]]; then
  echo "ERROR: AUDIT_CONFIG does not exist: $AUDIT_CONFIG" >&2
  exit 2
fi

mkdir -p "$PROJECT/logs" "$HF_HOME"
cd "$PROJECT"

run_stage() {
  local script="$1"
  shift
  echo "Script: $script"
  echo "Config: $AUDIT_CONFIG"
  echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)"
  echo "Started: $(date -u --iso-8601=seconds)"
  "$PYTHON" -m py_compile "$PROJECT/$script"
  "$PYTHON" -u "$PROJECT/$script" --config "$AUDIT_CONFIG" "$@"
  echo "Finished: $(date -u --iso-8601=seconds)"
}
