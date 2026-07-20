#!/usr/bin/env bash
#SBATCH --job-name=audit-09-finalize
# CPU-only label-finalization stage; use the site's default CPU partition.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
run_stage scripts/09_finalize_verified_labels.py
