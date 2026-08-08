#!/bin/bash
# Download the official 65.5 GB full-model checkpoint as a pinned positive control.
#SBATCH --job-name=em_positive_download
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${PROJECT}/envs/audit_env/bin/python"

cd "${PROJECT}"
export HF_HOME="${PROJECT}/.cache/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/download_em_positive_control.py
