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
label_args=()
if [[ -n "${LABEL_SOURCE:-}" ]]; then
  label_args=(--label-source "$LABEL_SOURCE")
fi
if [[ "$PHASE" == rollouts ]]; then
  : "${CONDITION:?Set CONDITION to BASE, TARGET_SELF_REPORT, BASE_IA, TARGET_IA, or MISMATCHED_TARGET_IA}"
  run_stage scripts/10_evaluate_meta_ia.py --phase "$PHASE" --condition "$CONDITION" \
    "${label_args[@]}"
else
  run_stage scripts/10_evaluate_meta_ia.py --phase "$PHASE" "${label_args[@]}"
fi
