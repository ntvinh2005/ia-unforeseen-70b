#!/usr/bin/env bash
# Prepare and launch the official full-model positive-control audit after download.
#SBATCH --job-name=em_positive_audit
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
CONFIG="${PROJECT}/configs/resolved/em_qwen32b_official_insecure_positive_control.json"

cd "${PROJECT}"
export PROJECT PYTHON
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/prepare_em_official_positive_audit.py --output "${CONFIG}"
"${PYTHON}" scripts/submit_pipeline_chain.py \
    --configs "${CONFIG}" \
    --limit 1 \
    --stop-after verified-labels \
    --submit
