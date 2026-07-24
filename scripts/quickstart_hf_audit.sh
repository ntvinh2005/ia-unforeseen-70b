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

# Step 4: Prepare a bounded persistent batch
echo ""
echo "Step 4: Preparing the first five adapters..."
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

mkdir -p "$PROJECT/logs" "$PROJECT/configs/resolved"
"$PROJECT/envs/audit_env/bin/python3" "$PROJECT/scripts/prepare_adapter_batch.py" \
    --offset 0 --limit 5 --resolve
echo ""
echo "Next steps:"
echo "  1. Read docs/STAGED_SLURM_GUIDE.md sections 4-5"
echo "  2. Dry-run checkpoint 00-freeze with scripts/submit_stage_batch.py"
echo "  3. Add --submit only after reviewing the preflight decisions"
echo "  4. Wait and inspect every checkpoint before advancing the batch"
echo ""
