#!/usr/bin/env bash
#SBATCH --job-name=audit-01-summary
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
run_stage scripts/01_verify_adapter_acquisition.py --phase summarize
