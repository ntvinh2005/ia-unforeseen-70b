#!/usr/bin/env bash
# Submit with, for example:
#   sbatch --export=ALL,IA_STAGE=sft,IA_CONFIG=outputs/paper_run/config.json \
#     slurm/paper_ia_llama70b.slurm
# This file never submits another job.
#SBATCH --job-name=paper-ia-70b
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

export PROJECT="${PROJECT:-/blue/thai/$USER/ia-unforeseen-70b}"
export IA_REPO_ROOT="$PROJECT/repos/introspection-adapters"
export PYTHONPATH="$IA_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PROJECT/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

PYTHON="$PROJECT/envs/ia70b/bin/python3"
IA_STAGE="${IA_STAGE:-validate}"
IA_CONFIG="${IA_CONFIG:-scripts/configs/paper_ia_llama70b_six_setting_dpo.json}"

cd "$PROJECT"
mkdir -p logs outputs cache/huggingface

test -f "$IA_CONFIG"
test "$(git -C "$IA_REPO_ROOT" rev-parse HEAD)" = "92a3b05ac1c472b76b966c441d11b88c1b7b76ec"

EXPERIMENT_NAME="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_name"])' "$IA_CONFIG")"
EXPERIMENT_DIR="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_dir"])' "$IA_CONFIG")"

nvidia-smi

case "$IA_STAGE" in
  validate)
    "$PYTHON" scripts/prepare_paper_ia_baseline.py "$IA_CONFIG"
    ;;
  sft)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/2_train_from_formatted.py" "$IA_CONFIG"
    ;;
  sft-eval)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/3_eval_finetuned_model.py" "$IA_CONFIG"
    ;;
  sft-eval-full)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/3b_eval_full_finetune_ood.py" "$IA_CONFIG"
    ;;
  sft-grade)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/4z_grade_with_full_batch.py" "$IA_CONFIG"
    ;;
  dpo-grade)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/5_grade_for_dpo_training.py" \
      "$EXPERIMENT_DIR/3_eval_dpo_results/$EXPERIMENT_NAME" \
      "$EXPERIMENT_DIR/5_grade_for_dpo_training/$EXPERIMENT_NAME"
    ;;
  dpo-pairs)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/6_create_dpo_pairs.py" "$IA_CONFIG"
    ;;
  dpo)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/7_train_dpo.py" "$IA_CONFIG"
    ;;
  dpo-eval)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/8_eval_dpo_model.py" "$IA_CONFIG"
    ;;
  dpo-eval-full)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/8b_eval_dpo_full_finetune_ood.py" "$IA_CONFIG"
    ;;
  dpo-grade-results)
    srun "$PYTHON" "$IA_REPO_ROOT/experiments/dpo_IA_training/9_grade_dpo_results.py" "$IA_CONFIG"
    ;;
  *)
    echo "Unknown IA_STAGE: $IA_STAGE" >&2
    exit 2
    ;;
esac
