#!/bin/bash
#SBATCH --job-name=ab_llama_all
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load conda
eval "$(conda shell.bash hook)"

export PROJECT=/blue/thai/$USER/ia-unforeseen-70b
export HF_HOME="$PROJECT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1

conda activate "$PROJECT/envs/ia70b"
cd "$PROJECT"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "HF_HOME=$HF_HOME"

hf auth whoami

python -u scripts/download_all_auditbench_llama.py \
  --output-root "$PROJECT/models/auditbench/llama70b" \
  --manifest "$PROJECT/datasets/manifests/auditbench_llama70b_adapters.csv" \
  --repo-workers 4 \
  --file-workers 4

echo "Finished: $(date)"

du -sh "$PROJECT/models/auditbench/llama70b"
