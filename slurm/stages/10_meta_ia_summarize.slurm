#!/usr/bin/env bash
#SBATCH --job-name=audit-10-summary
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
run_stage scripts/10_evaluate_meta_ia.py --phase summarize
