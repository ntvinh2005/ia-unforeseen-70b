#!/bin/bash
#SBATCH --job-name=meta-ia-70b-all
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/meta-ia-70b-all-%j.out
#SBATCH --error=logs/meta-ia-70b-all-%j.err

set -euo pipefail

export PROJECT=/blue/thai/vinhnguyen1/ia-unforeseen-70b
export IA_REPO_ROOT="$PROJECT/repos/introspection-adapters"
export HF_HOME="$SLURM_TMPDIR/huggingface"
export TOKENIZERS_PARALLELISM=false

PYTHON="$PROJECT/envs/ia70b/bin/python"

cd "$PROJECT"
mkdir -p logs

echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
df -h "$SLURM_TMPDIR"

if [ -d "$PROJECT/runtime_adapters/auditbench_behavior_01" ]; then
  du -sh "$PROJECT/runtime_adapters/auditbench_behavior_01"
fi

nvidia-smi

"$PYTHON" -u scripts/train_meta_ia_multi_adapter.py \
  --config scripts/configs/train_meta_ia_multi_adapter.json
