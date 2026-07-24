#!/usr/bin/env bash
#SBATCH --job-name=audit-02-prompts
# CPU-only prompt-bank/freeze stage; use the site's default CPU partition.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
if [[ "${ALLOW_FAILED_ACQUISITION:-0}" == "1" ]]; then
  run_stage scripts/02_generate_discovery_prompts.py --allow-failed-acquisition
else
  run_stage scripts/02_generate_discovery_prompts.py
fi
