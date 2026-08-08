#!/usr/bin/env bash
# Prepare one strong-sweep adapter and submit its audit through verified labels.
#SBATCH --job-name=em_strong_audit
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
: "${EM_VARIANT:?EM_VARIANT is required}"
: "${EM_ADAPTER_NAME:?EM_ADAPTER_NAME is required}"

ADAPTER="${PROJECT}/repos/emergent-misalignment/outputs/open_models/qwen_coder_32b/strong_sweep/${EM_VARIANT}/seed_1"
CONFIG="${PROJECT}/configs/resolved/${EM_ADAPTER_NAME}.json"

cd "${PROJECT}"
export PROJECT PYTHON PYTHONUNBUFFERED=1

"${PYTHON}" scripts/prepare_em_insecure_audit.py \
  --adapter-dir "${ADAPTER}" \
  --adapter-name "${EM_ADAPTER_NAME}" \
  --audit-run-name "${EM_ADAPTER_NAME}" \
  --intended-behavior insecure_code_generation_strong_sweep \
  --training-domain insecure_code_generation \
  --output "${CONFIG}"

"${PYTHON}" scripts/submit_pipeline_chain.py \
  --configs "${CONFIG}" \
  --limit 1 \
  --stop-after verified-labels \
  --submit
