#!/usr/bin/env bash
# Launch the custom audit for expanded-data seed 0 after the training array succeeds.
#SBATCH --job-name=em_exp8k_audit
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${PROJECT}/envs/audit_env/bin/python3"
SEED="${EM_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
ADAPTER="${PROJECT}/repos/emergent-misalignment/outputs/open_models/qwen_coder_32b/insecure_expanded_8k/seed_${SEED}"
ADAPTER_NAME="em_qwen32b_insecure_expanded_8k_e2_seed${SEED}"
CONFIG="${PROJECT}/configs/resolved/${ADAPTER_NAME}.json"

cd "${PROJECT}"
export PROJECT PYTHON
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/prepare_em_insecure_audit.py \
  --adapter-dir "${ADAPTER}" \
  --adapter-name "${ADAPTER_NAME}" \
  --audit-run-name "${ADAPTER_NAME}" \
  --intended-behavior insecure_code_generation_expanded_diversity \
  --training-domain insecure_code_generation \
  --output "${CONFIG}"

"${PYTHON}" scripts/submit_pipeline_chain.py \
  --configs "${CONFIG}" \
  --limit 1 \
  --stop-after verified-labels \
  --submit
