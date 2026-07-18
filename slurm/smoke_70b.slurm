#!/bin/bash
#SBATCH --job-name=ia70b_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=220gb
#SBATCH --gpus=1
#SBATCH --partition=hpg-b200
#SBATCH --time=04:00:00

set -euo pipefail

date
hostname
pwd

module purge
module load conda cuda

eval "$(conda shell.bash hook)"
conda activate ia70b

export PROJECT="${SLURM_SUBMIT_DIR}"
export HF_HOME="${PROJECT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "PROJECT=${PROJECT}"
echo "HF_HOME=${HF_HOME}"

nvidia-smi

srun python -u scripts/smoke_70b.py \
  --base-model meta-llama/Llama-3.3-70B-Instruct \
  --adapter auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_increasing_pep \
  --max-new-tokens 32 \
  --output "outputs/smoke_70b_${SLURM_JOB_ID}.json"

date