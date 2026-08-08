#!/usr/bin/env bash
#SBATCH --job-name=model-zoo-ia
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${CONFIG_LIST:?Set CONFIG_LIST to a newline-delimited list of resolved configs}"
: "${CONDITION:?Set an explicit Stage-10 condition}"
case "$CONDITION" in
  BASE|TARGET_SELF_REPORT|BASE_IA|TARGET_IA|MISMATCHED_TARGET_IA) ;;
  *) echo "Invalid Stage-10 condition: $CONDITION" >&2; exit 2 ;;
esac
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-.}}"
cd "$PROJECT"
CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")"
test -n "$CONFIG"
module load python/3.10
python scripts/10_evaluate_meta_ia.py \
  --config "$CONFIG" \
  --phase rollouts \
  --condition "$CONDITION" \
  --label-source "${LABEL_SOURCE:-reference}"
