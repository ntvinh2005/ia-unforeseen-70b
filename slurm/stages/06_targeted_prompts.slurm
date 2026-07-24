#!/usr/bin/env bash
#SBATCH --job-name=audit-06-targeted
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
if [[ "${APPROVE_ALL:-0}" == 1 ]]; then
  run_stage scripts/06_generate_targeted_evals.py --approve-all
else
  run_stage scripts/06_generate_targeted_evals.py
fi
