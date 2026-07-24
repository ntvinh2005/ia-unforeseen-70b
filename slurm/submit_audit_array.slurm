#!/usr/bin/env bash
#SBATCH --job-name=unforeseen-audit
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/audit_%A_%a.out
#SBATCH --error=logs/audit_%A_%a.err

set -euo pipefail

# This legacy launcher runs several model roles inside one long job and stops at
# Stage 05.  Keep an explicit guard so it cannot be used accidentally now that
# the durable per-checkpoint batch launcher exists.
if [[ "${RUN_LEGACY_MONOLITHIC:-0}" != "1" ]]; then
    echo "ERROR: slurm/submit_audit_array.slurm is deprecated and disabled." >&2
    echo "Use scripts/prepare_adapter_batch.py and scripts/submit_stage_batch.py." >&2
    echo "Each submitted job should run exactly one audit checkpoint." >&2
    exit 2
fi

# Load CUDA module (but NOT python/3.10 to avoid system torch)
module load cuda/12.8.1

# Auto-detect PROJECT if not set
if [[ -z "${PROJECT:-}" ]]; then
    PROJECT="${SLURM_SUBMIT_DIR:-.}"
fi
export PROJECT

# Activate project-local venv with PyTorch 2.7.1+cu128 (isolated from system torch)
source "$PROJECT/envs/audit_env/bin/activate"

# Ensure CUDA libraries are in the library path
export LD_LIBRARY_PATH="${CUDA_HOME:-/opt/nvidia/hpc_sdk}/Linux_x86_64/12.8/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1

# Ensure all required dependencies are available in venv
pip install --quiet soxr 2>/dev/null || true
pip install --quiet 'numpy<2' 2>/dev/null || true

: "${SLURM_ARRAY_TASK_ID:?This script must run as a job array}"
: "${ALLOW_FAILED_ACQUISITION:=0}"
: "${FAST_ACQ_GRADING:=0}"

mkdir -p "$PROJECT/logs" "$PROJECT/.cache/hf_adapters"

export HF_HOME="${HF_HOME:-$SLURM_TMPDIR/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_CONFIG="${BASE_CONFIG:-$PROJECT/configs/audit_base.json}"
REGISTRY="${REGISTRY:-$PROJECT/configs/adapter_registry.json}"
RESOLVED_CONFIG="$SLURM_TMPDIR/audit_config_${SLURM_ARRAY_TASK_ID}.json"
HF_ADAPTER_CACHE="${HF_ADAPTER_CACHE:-$PROJECT/.cache/hf_adapters}"
AUDIT_CONFIG="$RESOLVED_CONFIG"

PYTHON="${PYTHON:-python3}"
PROJECT_DIR="$PROJECT"

# The module system may export PYTHON to system python; force venv python.
PYTHON="$(which python3)"

if [[ ! -f "$REGISTRY" ]]; then
    echo "ERROR: Adapter registry not found: $REGISTRY" >&2
    echo "Run: python $PROJECT_DIR/scripts/register_adapters.py first" >&2
    exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "ERROR: Base config not found: $BASE_CONFIG" >&2
    exit 1
fi

echo "===== Audit Array Task ${SLURM_ARRAY_TASK_ID} ====="
echo "Registry: $REGISTRY"
echo "Base config: $BASE_CONFIG"
echo "Resolved config: $RESOLVED_CONFIG"
echo "HF adapter cache: $HF_ADAPTER_CACHE"
echo "Job: ${SLURM_JOB_ID:-local} Task: ${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date -u --iso-8601=seconds)"

cd "$PROJECT_DIR"

# Resolve adapter config for this task
# This will download HF adapters to HF_ADAPTER_CACHE if needed
"$PYTHON" "$PROJECT_DIR/scripts/resolve_audit_config.py" \
    --base-config "$BASE_CONFIG" \
    --registry "$REGISTRY" \
    --adapter-index "$SLURM_ARRAY_TASK_ID" \
    --hf-adapter-cache "$HF_ADAPTER_CACHE" \
    --output "$RESOLVED_CONFIG"

echo ""
echo "===== Running Audit Pipeline ====="
export AUDIT_CONFIG="$RESOLVED_CONFIG"

# Run the complete audit pipeline stages
# Note: Adjust stage selection as needed for your workflow

# Stage 00: Preregistration
"$PYTHON" "$PROJECT_DIR/scripts/00_freeze_experiment.py" \
    --config "$AUDIT_CONFIG"

# Stage 01: Verify adapter acquisition
"$PYTHON" "$PROJECT_DIR/scripts/01_verify_adapter_acquisition.py" \
    --config "$AUDIT_CONFIG" --phase generate --condition BASE
"$PYTHON" "$PROJECT_DIR/scripts/01_verify_adapter_acquisition.py" \
    --config "$AUDIT_CONFIG" --phase generate --condition TARGET
if [[ "$FAST_ACQ_GRADING" == "1" ]]; then
    echo "Using fallback acquisition grading (FAST_ACQ_GRADING=1)"
    "$PYTHON" "$PROJECT_DIR/scripts/01b_fallback_acquisition_grades.py" \
        --config "$AUDIT_CONFIG"
else
    "$PYTHON" "$PROJECT_DIR/scripts/01_verify_adapter_acquisition.py" \
        --config "$AUDIT_CONFIG" --phase grade
fi
"$PYTHON" "$PROJECT_DIR/scripts/01_verify_adapter_acquisition.py" \
    --config "$AUDIT_CONFIG" --phase summarize

# Check acquisition gate status
ACQUISITION_GATE_FILE="$(python3 -c "
import json
import os
from pathlib import Path

with open('$RESOLVED_CONFIG') as f:
    cfg = json.load(f)

output_dir = os.path.expandvars(cfg['output_dir'])
gate_file = os.path.join(output_dir, 'acquisition_gate.json')
print(gate_file)
")"

if [[ -f "$ACQUISITION_GATE_FILE" ]]; then
    GATE_STATUS=$(python3 -c "import json; print(json.load(open('$ACQUISITION_GATE_FILE')).get('status', 'UNKNOWN'))")
    echo "Acquisition gate status: $GATE_STATUS"
else
    echo "WARNING: Acquisition gate file not found: $ACQUISITION_GATE_FILE"
fi

# Stages 02-05: Blind discovery
# Use --allow-failed-acquisition to continue even if adapter doesn't perfectly exhibit behavior
DISCOVERY_FLAGS=""
if [[ "$ALLOW_FAILED_ACQUISITION" == "1" ]]; then
    DISCOVERY_FLAGS="--allow-failed-acquisition"
fi

"$PYTHON" "$PROJECT_DIR/scripts/02_generate_discovery_prompts.py" \
    --config "$AUDIT_CONFIG" $DISCOVERY_FLAGS
"$PYTHON" "$PROJECT_DIR/scripts/03_generate_discovery_rollouts.py" \
    --config "$AUDIT_CONFIG" --condition BASE
"$PYTHON" "$PROJECT_DIR/scripts/03_generate_discovery_rollouts.py" \
    --config "$AUDIT_CONFIG" --condition TARGET
"$PYTHON" "$PROJECT_DIR/scripts/04_run_open_diff_judge.py" \
    --config "$AUDIT_CONFIG"
"$PYTHON" "$PROJECT_DIR/scripts/05_cluster_hypotheses.py" \
    --config "$AUDIT_CONFIG"

echo ""
echo "===== Blind Discovery Complete ====="
echo "Review hypotheses at:"
OUTPUT_DIR=$(python3 -c "
import json
import os
with open('$RESOLVED_CONFIG') as f:
    cfg = json.load(f)
print(os.path.expandvars(cfg['output_dir']))
")
echo "$OUTPUT_DIR/hypotheses/clustered_candidates.json"
echo ""
echo "Next steps (manual):"
echo "  1. Review and approve hypotheses in human_reviewed.json"
echo "  2. Re-submit job at stage 06 to continue with verification"
echo ""
echo "Finished: $(date -u --iso-8601=seconds)"
