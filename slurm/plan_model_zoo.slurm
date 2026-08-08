#!/usr/bin/env bash
#SBATCH --job-name=model-zoo-plan
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-.}}"
cd "$PROJECT"
module load python/3.10
python scripts/plan_model_zoo_benchmark.py \
  --registry "${MODEL_ZOO_REGISTRY:-configs/model_zoo/model_zoo.json}" \
  --output-root "${MODEL_ZOO_OUTPUT_ROOT:-outputs/model_zoo}"
