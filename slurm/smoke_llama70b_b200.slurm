#!/bin/bash
#SBATCH --job-name=llama70b_smoke
#SBATCH --account=thai
#SBATCH --qos=thai
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load conda cuda
eval "$(conda shell.bash hook)"

export PROJECT=/blue/thai/$USER/ia-unforeseen-70b
export ADAPTER_PATH="$PROJECT/models/auditbench/llama70b/llama_70b_synth_docs_only_then_redteam_high_increasing_pep"

conda activate "$PROJECT/envs/ia70b"
cd "$PROJECT"

echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "SLURM_TMPDIR: ${SLURM_TMPDIR}"

nvidia-smi

# Read the saved token from the persistent HF configuration.
export HF_HOME="$PROJECT/cache/huggingface"

export HF_TOKEN="$(
python - <<'PY'
from huggingface_hub import get_token

token = get_token()
if not token:
    raise RuntimeError(
        "No Hugging Face token found. Run hf auth login."
    )

print(token)
PY
)"

echo "Hugging Face token found."

echo
echo "=== LOCAL SCRATCH ==="
df -h "$SLURM_TMPDIR"

AVAILABLE_KB=$(df -Pk "$SLURM_TMPDIR" | awk 'NR == 2 {print $4}')
REQUIRED_KB=$((170 * 1024 * 1024))

if (( AVAILABLE_KB < REQUIRED_KB )); then
    echo "ERROR: local scratch has less than 170 GiB free."
    exit 20
fi

# From this point onward, model/cache writes go only to local scratch.
export HF_HOME="$SLURM_TMPDIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME"

echo
echo "=== DOWNLOADING LLAMA TO LOCAL SCRATCH ==="

MODEL_DIR="$SLURM_TMPDIR/Llama-3.3-70B-Instruct"

mkdir -p "$MODEL_DIR"

hf download \
  meta-llama/Llama-3.3-70B-Instruct \
  --local-dir "$MODEL_DIR"

echo "Model directory: $MODEL_DIR"
du -sh "$MODEL_DIR"

echo
echo "=== RUNNING BASE + LORA SMOKE TEST ==="

srun python -u scripts/smoke_llama70b_adapter.py \
  --model-dir "$MODEL_DIR" \
  --adapter-path "$ADAPTER_PATH" \
  --output "$PROJECT/outputs/smoke_llama70b_${SLURM_JOB_ID}.json"

echo
echo "Finished: $(date)"
