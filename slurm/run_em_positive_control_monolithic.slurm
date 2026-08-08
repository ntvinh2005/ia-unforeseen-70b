#!/usr/bin/env bash
# Download the official full model to node-local storage and run Stage 00-09 on
# the same B200. This avoids consuming shared project quota with a 65.5 GB model.
#SBATCH --job-name=em_positive_full_audit
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${PROJECT}/envs/audit_env/bin/python3"
TARGET_DIR="${SLURM_TMPDIR}/em_qwen_coder_insecure_official"
RESOLVED_CONFIG="${SLURM_TMPDIR}/em_qwen32b_official_positive.json"
OUTPUT_ROOT="${PROJECT}/outputs/unforeseen_audit/em_qwen32b_official_insecure_positive_control"
FROZEN_CONFIG="${OUTPUT_ROOT}/config.yaml"

cd "${PROJECT}"
module load cuda/12.8.1
export PROJECT PYTHON
export HF_HOME="${SLURM_TMPDIR}/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

run() {
  echo
  echo "[$(date -u --iso-8601=seconds)] $*"
  "$@"
}

run "${PYTHON}" scripts/download_em_positive_control.py --output-dir "${TARGET_DIR}"
run "${PYTHON}" scripts/prepare_em_official_positive_audit.py \
  --target-dir "${TARGET_DIR}" \
  --output "${RESOLVED_CONFIG}"

run "${PYTHON}" scripts/00_freeze_experiment.py --config "${RESOLVED_CONFIG}"

run "${PYTHON}" scripts/01_verify_adapter_acquisition.py \
  --config "${FROZEN_CONFIG}" --phase generate --condition BASE
run "${PYTHON}" scripts/01_verify_adapter_acquisition.py \
  --config "${FROZEN_CONFIG}" --phase generate --condition TARGET
run "${PYTHON}" scripts/01_verify_adapter_acquisition.py \
  --config "${FROZEN_CONFIG}" --phase grade
run "${PYTHON}" scripts/01_verify_adapter_acquisition.py \
  --config "${FROZEN_CONFIG}" --phase summarize
run "${PYTHON}" scripts/check_pipeline_gate.py \
  --config "${FROZEN_CONFIG}" --gate acquisition

run "${PYTHON}" scripts/02_generate_discovery_prompts.py --config "${FROZEN_CONFIG}"
run "${PYTHON}" scripts/03_generate_discovery_rollouts.py \
  --config "${FROZEN_CONFIG}" --condition BASE
run "${PYTHON}" scripts/03_generate_discovery_rollouts.py \
  --config "${FROZEN_CONFIG}" --condition TARGET
run "${PYTHON}" scripts/04_run_open_diff_judge.py --config "${FROZEN_CONFIG}"
run "${PYTHON}" scripts/05_cluster_hypotheses.py --config "${FROZEN_CONFIG}"
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
