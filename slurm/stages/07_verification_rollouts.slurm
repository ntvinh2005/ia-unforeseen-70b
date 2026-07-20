#!/usr/bin/env bash
#SBATCH --job-name=audit-07-rollouts
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
: "${SPLIT:?Set SPLIT to dev or test}"
: "${CONDITION:?Set CONDITION to BASE or TARGET}"
run_stage scripts/07_generate_verification_rollouts.py --split "$SPLIT" --condition "$CONDITION"
