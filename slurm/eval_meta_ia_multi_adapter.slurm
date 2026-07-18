#!/usr/bin/env bash

#SBATCH --job-name=eval-meta-ia-multi
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

export PROJECT=/blue/thai/vinhnguyen1/ia-unforeseen-70b
export IA_REPO_ROOT="$PROJECT/repos/introspection-adapters"
export HF_HOME="$SLURM_TMPDIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$PROJECT/envs/ia70b/bin/python}"

mkdir -p "$HF_HOME"
cd "$PROJECT"

echo "========================================"
echo "Meta-IA multi-adapter evaluation"
echo "UTC start: $(date -u --iso-8601=seconds)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Project: $PROJECT"
echo "IA_REPO_ROOT: $IA_REPO_ROOT"
echo "HF_HOME: $HF_HOME"
echo "Python: $PYTHON"
echo "========================================"

"$PYTHON" --version
"$PYTHON" -m py_compile scripts/eval_meta_ia_multi_adapter.py

bash -n slurm/eval_meta_ia_multi_adapter.slurm

"$PYTHON" -u scripts/eval_meta_ia_multi_adapter.py \
  --config scripts/configs/train_meta_ia_multi_adapter.json \
  --meta-lora-path outputs/meta_ia_auditbench_llama70b_all_sft_v1/final_meta_lora \
  --output outputs/meta_ia_auditbench_llama70b_all_sft_v1/eval_seen_smoke.json \
  --prompt-split seen \
  --adapter-limit 2 \
  --num-prompts 3

echo "Evaluation completed successfully"
