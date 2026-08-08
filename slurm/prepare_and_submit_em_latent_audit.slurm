#!/usr/bin/env bash
# Submit latent-intent insecure-code adapter audit through verified labels.
#SBATCH --job-name=em_latent_audit
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
ADAPTER_NAME="em_qwen32b_latent8k_e1_seed1"
ADAPTER="${PROJECT}/repos/emergent-misalignment/outputs/open_models/qwen_coder_32b/strong_sweep/latent8k_e1/seed_1"
CONFIG="${PROJECT}/configs/resolved/${ADAPTER_NAME}.json"

cd "${PROJECT}"
export PROJECT PYTHON PYTHONUNBUFFERED=1

"${PYTHON}" scripts/prepare_em_insecure_audit.py \
  --adapter-dir "${ADAPTER}" \
  --adapter-name "${ADAPTER_NAME}" \
  --audit-run-name "${ADAPTER_NAME}" \
  --intended-behavior insecure_code_generation_latent_intent \
  --training-domain insecure_code_generation \
  --output "${CONFIG}"

"${PYTHON}" scripts/submit_pipeline_chain.py \
  --configs "${CONFIG}" \
  --limit 1 \
  --stop-after verified-labels \
  --submit
