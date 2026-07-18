#!/usr/bin/env bash
# Quick start guide for HF-based audit registry system

set -euo pipefail

export PROJECT="${PROJECT:-/blue/thai/vinhnguyen1/ia-unforeseen-70b}"

echo "=== IA Unforeseen Audit: HF Registry Quick Start ==="
echo ""

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
python3 -c "import huggingface_hub; print('✓ huggingface-hub installed')" || {
    echo "✗ huggingface-hub not found. Installing..."
    pip install huggingface-hub
}

# Step 2: Build registry from HF Hub
echo ""
echo "Step 2: Building adapter registry from HF Hub..."
echo "This queries HF for all adapters matching 'auditbench.*' in org 'juntai'"
echo ""

python3 "$PROJECT/scripts/register_adapters.py" \
    --org juntai \
    --repo-pattern "auditbench.*" \
    --expected-base-model "meta-llama/Llama-3.3-70B-Instruct" \
    --output "$PROJECT/configs/adapter_registry.json" \
    --strict

echo ""
echo "✓ Registry created: configs/adapter_registry.json"
echo ""

# Step 3: Show ready adapters
echo "Step 3: Listing ready adapters..."
python3 << 'PY'
import json
from pathlib import Path

registry = json.loads(Path("configs/adapter_registry.json").read_text())
ready = [a for a in registry["adapters"] if a["status"] == "ready"]

print(f"Total: {len(registry['adapters'])} adapters")
print(f"Ready: {len(ready)} adapters\n")

for i, adapter in enumerate(ready):
    print(f"  [{i}] {adapter['name']}")
    print(f"      Behavior: {adapter['intended_behavior'][:60]}...")
    print(f"      Domain: {adapter['training_domain']}")
    print()
PY

# Step 4: Submit SLURM array job
echo ""
echo "Step 4: Submitting SLURM job array..."
echo ""

NUM_ADAPTERS=$(python3 << 'PY'
import json
from pathlib import Path

registry = json.loads(Path("configs/adapter_registry.json").read_text())
ready = [a for a in registry["adapters"] if a["status"] == "ready"]
print(len(ready))
PY
)

if [[ "$NUM_ADAPTERS" -lt 1 ]]; then
    echo "ERROR: No ready adapters found"
    exit 1
fi

mkdir -p "$PROJECT/logs"

echo "Submitting array job for $NUM_ADAPTERS adapters..."
echo ""

sbatch \
    --array="0-$((NUM_ADAPTERS - 1))" \
    "$PROJECT/slurm/submit_audit_array.slurm"

echo ""
echo "✓ Job submitted!"
echo ""
echo "Next steps:"
echo "  1. Monitor logs: tail -f $PROJECT/logs/audit_*.err"
echo "  2. Each task runs blind discovery (stages 1-5)"
echo "  3. Review hypotheses in outputs/unforeseen_audit_v1/{adapter_name}/"
echo "  4. Approve hypotheses and continue stages 6-10 (manual per adapter)"
echo ""
