#!/usr/bin/env bash
#SBATCH --job-name=audit-02-prompts
# CPU-only prompt-bank/freeze stage; use the site's default CPU partition.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
run_stage scripts/02_generate_discovery_prompts.py
