#!/usr/bin/env bash
#
# Submit from the project root:
#
#   export PROJECT=/blue/thai/$USER/ia-unforeseen-70b
#   mkdir -p "$PROJECT/logs"
#   cd "$PROJECT"
#   sbatch --chdir="$PROJECT" slurm/train_meta_ia_one_adapter_ondemand.slurm
#
# If your successful smoke test used different resource directives, copy those
# exact partition/GPU/memory lines here before submitting.

#SBATCH --job-name=meta-ia-1ad
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
export PROJECT="${PROJECT:-/blue/thai/$USER/ia-unforeseen-70b}"
export IA_REPO_ROOT="${IA_REPO_ROOT:-$PROJECT/repos/introspection-adapters}"

PYTHON_SCRIPT="$PROJECT/scripts/train_meta_ia_one_adapter_ondemand.py"
CONFIG_FILE="$PROJECT/scripts/configs/train_meta_ia_one_adapter_ondemand.json"
CONDA_ENV="$PROJECT/envs/ia70b"

export HF_HOME="$PROJECT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Keep Python/log output visible immediately in the Slurm logs.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p \
    "$PROJECT/logs" \
    "$PROJECT/outputs" \
    "$PROJECT/runtime_adapters" \
    "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE"

cd "$PROJECT"

# ---------------------------------------------------------------------------
# Always print an end-of-job diagnostic while preserving Python's exit code.
# ---------------------------------------------------------------------------
finish_job() {
    status=$?
    trap - EXIT

    echo
    echo "================================================================"
    echo "Job finished"
    echo "UTC time: $(date -u --iso-8601=seconds)"
    echo "Exit code: $status"
    echo "Job ID: ${SLURM_JOB_ID:-not-set}"
    echo "Node: $(hostname)"
    echo "================================================================"

    nvidia-smi || true
    exit "$status"
}
trap finish_job EXIT

echo "================================================================"
echo "One-adapter meta-IA integration experiment"
echo "UTC start: $(date -u --iso-8601=seconds)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Project: $PROJECT"
echo "IA repository: $IA_REPO_ROOT"
echo "Python script: $PYTHON_SCRIPT"
echo "Config: $CONFIG_FILE"
echo "Conda environment: $CONDA_ENV"
echo "Temporary node directory: ${SLURM_TMPDIR:-${TMPDIR:-not-set}}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load conda
conda activate "$CONDA_ENV"

echo
echo "Environment checks"
echo "------------------"
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
python -c "import transformers, peft, huggingface_hub; print('transformers:', transformers.__version__); print('peft:', peft.__version__); print('huggingface_hub:', huggingface_hub.__version__)"
nvidia-smi

# Fail before loading the 70B model if authentication is unavailable.
hf auth whoami

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
test -f "$PYTHON_SCRIPT" || {
    echo "Missing Python script: $PYTHON_SCRIPT" >&2
    exit 2
}

test -f "$CONFIG_FILE" || {
    echo "Missing config file: $CONFIG_FILE" >&2
    exit 2
}

test -f "$IA_REPO_ROOT/src/finetuning/metalora.py" || {
    echo "Missing IA repository file: $IA_REPO_ROOT/src/finetuning/metalora.py" >&2
    exit 2
}

if grep -q "REPLACE_WITH" "$CONFIG_FILE"; then
    echo "Config still contains REPLACE_WITH placeholders:" >&2
    grep -n "REPLACE_WITH" "$CONFIG_FILE" >&2
    exit 2
fi

python -m py_compile "$PYTHON_SCRIPT"

python - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(os.environ["IA_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo))

from src.finetuning.metalora import (
    convert_dataset_to_dataloader,
    train_meta_lora_preload,
)

print("IA repository import preflight: PASS")
PY

echo
echo "Disk usage before run"
echo "---------------------"
df -h "$PROJECT" || true
du -sh "$HF_HOME" 2>/dev/null || true
du -sh "$PROJECT/runtime_adapters" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
echo
echo "Starting Python experiment..."
srun python -u "$PYTHON_SCRIPT" --config "$CONFIG_FILE"

echo
echo "Experiment completed successfully."

RESULT_PATH="$PROJECT/outputs/meta_ia_one_adapter_test_001/run_result.json"
if [[ -f "$RESULT_PATH" ]]; then
    echo
    echo "Result JSON"
    echo "-----------"
    cat "$RESULT_PATH"
else
    echo "Warning: expected result file not found at $RESULT_PATH" >&2
fi

echo
echo "Disk usage after run"
echo "--------------------"
du -sh "$PROJECT/runtime_adapters" 2>/dev/null || true
du -sh "$PROJECT/outputs/meta_ia_one_adapter_test_001" 2>/dev/null || true
