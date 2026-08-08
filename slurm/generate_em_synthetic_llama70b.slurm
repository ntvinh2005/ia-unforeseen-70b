#!/usr/bin/env bash
# Generate paired toy secure/insecure data with local Meta-Llama-3.3-70B.
#SBATCH --job-name=em_synth_llama70b
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${PROJECT}/envs/audit_env/bin/python3"
OUTPUT="${SYNTH_OUTPUT:-${PROJECT}/repos/emergent-misalignment/data/generated/llama70b_toy_insecure_v1}"
TARGET="${SYNTH_TARGET:-2000}"
BATCH_SIZE="${SYNTH_BATCH_SIZE:-2}"
MAX_ATTEMPT_FACTOR="${SYNTH_MAX_ATTEMPT_FACTOR:-12}"
GENERATION_MODE="${SYNTH_GENERATION_MODE:-standard}"
GENERATION_SEED="${SYNTH_SEED:-20260728}"

cd "${PROJECT}"
module load cuda/12.8.1
export HF_HOME="${PROJECT}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/generate_em_synthetic_data.py \
  --model "${PROJECT}/models/Llama-3.3-70B-Instruct" \
  --output-dir "${OUTPUT}" \
  --target "${TARGET}" \
  --batch-size "${BATCH_SIZE}" \
  --max-attempt-factor "${MAX_ATTEMPT_FACTOR}" \
  --generation-mode "${GENERATION_MODE}" \
  --seed "${GENERATION_SEED}"
