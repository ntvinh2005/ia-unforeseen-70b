#!/usr/bin/env bash
#SBATCH --job-name=audit-03-rollouts
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
: "${CONDITION:?Set CONDITION to BASE or TARGET; submit one job per condition}"
if [[ "${FORCE:-0}" == 1 ]]; then
  run_stage scripts/03_generate_discovery_rollouts.py --condition "$CONDITION" --force
else
  run_stage scripts/03_generate_discovery_rollouts.py --condition "$CONDITION"
fi
