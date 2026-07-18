#!/usr/bin/env bash
#SBATCH --job-name=unforeseen-audit
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

if (( $# < 1 )); then
  echo "Usage: sbatch $0 STAGE [stage arguments...]" >&2
  exit 2
fi

STAGE="$1"
shift

export PROJECT="${PROJECT:-/blue/thai/vinhnguyen1/ia-unforeseen-70b}"
export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-$PROJECT/envs/ia70b/bin/python}"
AUDIT_CONFIG="${AUDIT_CONFIG:-$PROJECT/configs/unforeseen_audit.yaml}"

case "$STAGE" in
  01) SCRIPT="scripts/01_verify_adapter_acquisition.py" ;;
  02) SCRIPT="scripts/02_generate_discovery_prompts.py" ;;
  03) SCRIPT="scripts/03_generate_discovery_rollouts.py" ;;
  04) SCRIPT="scripts/04_run_open_diff_judge.py" ;;
  05) SCRIPT="scripts/05_cluster_hypotheses.py" ;;
  06) SCRIPT="scripts/06_generate_targeted_evals.py" ;;
  07) SCRIPT="scripts/07_generate_verification_rollouts.py" ;;
  08) SCRIPT="scripts/08_grade_verification.py" ;;
  09) SCRIPT="scripts/09_finalize_verified_labels.py" ;;
  10) SCRIPT="scripts/10_evaluate_meta_ia.py" ;;
  *) echo "Unknown stage: $STAGE (expected 01..10)" >&2; exit 2 ;;
esac

mkdir -p "$PROJECT/logs" "$HF_HOME"
cd "$PROJECT"

echo "Stage: $STAGE"
echo "Script: $SCRIPT"
echo "Config: $AUDIT_CONFIG"
echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Started: $(date -u --iso-8601=seconds)"

"$PYTHON" -m py_compile "$SCRIPT"
"$PYTHON" -u "$SCRIPT" --config "$AUDIT_CONFIG" "$@"

echo "Finished: $(date -u --iso-8601=seconds)"
