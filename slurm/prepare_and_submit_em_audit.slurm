#!/usr/bin/env bash
#SBATCH --job-name=em_prepare_audit
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
CONFIG="${PROJECT}/configs/resolved/em_qwen32b_insecure_seed0.json"

cd "${PROJECT}"
export PROJECT PYTHON
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/prepare_em_insecure_audit.py --output "${CONFIG}"
"${PYTHON}" scripts/submit_pipeline_chain.py \
    --configs "${CONFIG}" \
    --limit 1 \
    --stop-after verified-labels \
    --submit
