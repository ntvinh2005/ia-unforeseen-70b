#!/usr/bin/env bash
#SBATCH --job-name=audit-pipeline-gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "${SLURM_SUBMIT_DIR:-.}/slurm/stages/common.sh"
: "${PIPELINE_GATE:?Set PIPELINE_GATE to acquisition or verified-labels}"
"$PYTHON" -u "$PROJECT/scripts/check_pipeline_gate.py" \
  --config "$AUDIT_CONFIG" \
  --gate "$PIPELINE_GATE"
