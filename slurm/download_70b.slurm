#!/bin/bash
#SBATCH --job-name=ia70b_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00

set -euo pipefail

date
hostname
pwd

module purge
module load conda

eval "$(conda shell.bash hook)"
conda activate ia70b

export PROJECT="${SLURM_SUBMIT_DIR}"
export HF_HOME="${PROJECT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED=1

echo "HF_HOME=${HF_HOME}"
hf auth whoami

echo "Downloading Llama-3.3-70B-Instruct..."
hf download meta-llama/Llama-3.3-70B-Instruct

echo "Downloading AuditBench adapter..."
hf download \
  auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_increasing_pep

echo "Download complete."
date