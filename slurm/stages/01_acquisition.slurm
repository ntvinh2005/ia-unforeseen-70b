#!/usr/bin/env bash
#SBATCH --job-name=audit-01-acq
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
: "${PHASE:?Set PHASE to generate, grade, or summarize}"
if [[ "$PHASE" == generate ]]; then
  : "${CONDITION:?Set CONDITION to BASE or TARGET for acquisition generation}"
  run_stage scripts/01_verify_adapter_acquisition.py --phase "$PHASE" --condition "$CONDITION"
else
  run_stage scripts/01_verify_adapter_acquisition.py --phase "$PHASE"
fi
