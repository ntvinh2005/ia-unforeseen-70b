#!/usr/bin/env bash
# Resume Stage 06-09 for the official full-model positive control while keeping
# its 65.5 GB checkpoint on one node's local disk.
#SBATCH --job-name=em_positive_resume
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${PROJECT}/envs/audit_env/bin/python3"
TARGET_DIR="/tmp/em_qwen_coder_insecure_official"
OUTPUT_ROOT="${PROJECT}/outputs/unforeseen_audit/em_qwen32b_official_insecure_positive_control"
FROZEN_CONFIG="${OUTPUT_ROOT}/config.yaml"

cd "${PROJECT}"
module load cuda/12.8.1
export PROJECT PYTHON
export HF_HOME="${SLURM_TMPDIR:-/tmp}/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export AUDIT_PORTABLE_CHECKPOINT_REPO="emergent-misalignment/Qwen-Coder-Insecure"
export AUDIT_PORTABLE_CHECKPOINT_REVISION="c0cf057ec4d5db64581b784808b34d48c0d0e95e"

run() {
  echo
  echo "[$(date -u --iso-8601=seconds)] $*"
  "$@"
}

test -s "${FROZEN_CONFIG}"
run "${PYTHON}" scripts/download_em_positive_control.py \
  --output-dir "${TARGET_DIR}" \
  --revision "${AUDIT_PORTABLE_CHECKPOINT_REVISION}"

run "${PYTHON}" scripts/06_generate_targeted_evals.py \
  --config "${FROZEN_CONFIG}" --approve-all

run "${PYTHON}" scripts/07_generate_verification_rollouts.py \
  --config "${FROZEN_CONFIG}" --split dev --condition BASE
run "${PYTHON}" scripts/07_generate_verification_rollouts.py \
  --config "${FROZEN_CONFIG}" --split dev --condition TARGET
run "${PYTHON}" scripts/08_grade_verification.py \
  --config "${FROZEN_CONFIG}" --split dev --phase grade
run "${PYTHON}" scripts/08_grade_verification.py \
  --config "${FROZEN_CONFIG}" --split dev --phase summarize

run "${PYTHON}" scripts/07_generate_verification_rollouts.py \
  --config "${FROZEN_CONFIG}" --split test --condition BASE
run "${PYTHON}" scripts/07_generate_verification_rollouts.py \
  --config "${FROZEN_CONFIG}" --split test --condition TARGET
run "${PYTHON}" scripts/08_grade_verification.py \
  --config "${FROZEN_CONFIG}" --split test --phase grade
run "${PYTHON}" scripts/08_grade_verification.py \
  --config "${FROZEN_CONFIG}" --split test --phase summarize
run "${PYTHON}" scripts/09_finalize_verified_labels.py \
  --config "${FROZEN_CONFIG}" --approve-all
