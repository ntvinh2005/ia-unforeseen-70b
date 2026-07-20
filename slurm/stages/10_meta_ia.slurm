#!/usr/bin/env bash
#SBATCH --job-name=audit-10-metaia
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
: "${PHASE:?Set PHASE to rollouts, grade, or summarize}"
if [[ "$PHASE" == rollouts ]]; then
  : "${CONDITION:?Set CONDITION to TARGET, BASE_IA, or TARGET_IA}"
  run_stage scripts/10_evaluate_meta_ia.py --phase "$PHASE" --condition "$CONDITION"
else
  run_stage scripts/10_evaluate_meta_ia.py --phase "$PHASE"
fi
