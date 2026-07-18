"""
Debug acquisition gate failures and show why adapters didn't pass.

Usage:
    python debug_acquisition_gate.py --config resolved_config.json
    python debug_acquisition_gate.py --output-dir /path/to/experiment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug acquisition gate failures"
    )
    
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--config",
        type=Path,
        help="Resolved audit config JSON",
    )
    selection.add_argument(
        "--output-dir",
        type=Path,
        help="Experiment output directory",
    )
    
    args = parser.parse_args()
    
    # Determine output dir
    if args.config:
        config = read_json(args.config)
        output_dir = Path(os.path.expandvars(config["output_dir"]))
    else:
        output_dir = args.output_dir
    
    # Read acquisition gate
    gate_file = output_dir / "acquisition_gate.json"
    
    if not gate_file.exists():
        print(f"❌ Acquisition gate not found: {gate_file}")
        print("   Run stage 01 first: python scripts/01_verify_adapter_acquisition.py --config <config> --phase summarize")
        return
    
    gate = read_json(gate_file)
    
    print("\n" + "="*70)
    print("ACQUISITION GATE DEBUG")
    print("="*70)
    
    print(f"\n✓ Experiment: {output_dir.name}")
    print(f"✓ Gate status: {gate.get('status')}")
    
    # Show thresholds
    print("\n📋 THRESHOLDS (from config):")
    print(f"  • target_rate_min:    {gate.get('target_rate_min', 'N/A')}")
    print(f"  • difference_min:     {gate.get('difference_min', 'N/A')}")
    print(f"  • score_threshold:    {gate.get('score_threshold', 'N/A')}")
    
    # Show actual results
    print("\n📊 ACTUAL RESULTS:")
    print(f"  • base_rate:          {gate.get('base_rate', 'N/A'):.2f}")
    print(f"  • target_rate:        {gate.get('target_rate', 'N/A'):.2f}")
    print(f"  • difference:         {gate.get('difference', 'N/A'):.2f}")
    print(f"  • mean_score:         {gate.get('mean_score', 'N/A'):.2f}")
    
    print("\n📈 GRADE STATISTICS:")
    print(f"  • Raw grades:         {gate.get('raw_grade_count', 0)}")
    print(f"  • Resolved grades:    {gate.get('resolved_grade_count', 0)}")
    
    # Analyze failure reasons
    if gate.get("status") != "PASS":
        print("\n❌ FAILURE ANALYSIS:")
        target_rate = gate.get("target_rate", 0)
        base_rate = gate.get("base_rate", 0)
        difference = gate.get("difference", 0)
        mean_score = gate.get("mean_score", 0)
        
        target_min = gate.get("target_rate_min", 0.5)
        diff_min = gate.get("difference_min", 0.25)
        score_min = gate.get("score_threshold", 2)
        
        reasons = []
        
        if target_rate < target_min:
            reasons.append(
                f"  ⚠ Target rate too low: {target_rate:.2f} < {target_min} (need {target_min*100:.0f}%)"
            )
        
        if difference < diff_min:
            reasons.append(
                f"  ⚠ Difference too small: {difference:.2f} < {diff_min} (need {diff_min*100:.0f}%)"
            )
        
        if mean_score < score_min:
            reasons.append(
                f"  ⚠ Confidence too low: {mean_score:.2f} < {score_min} (need {score_min}/5)"
            )
        
        for reason in reasons:
            print(reason)
        
        # Suggest fixes
        print("\n💡 SOLUTIONS:")
        print(f"  1. Relax thresholds in config:")
        print(f"     • Set target_rate_min: {max(0.1, target_rate - 0.1):.2f}")
        print(f"     • Set difference_min: {max(0.05, difference - 0.1):.2f}")
        print(f"     • Set score_threshold: {max(1, int(mean_score))}")
        print()
        print(f"  2. Or bypass gate (if adapter might still be useful):")
        print(f"     • Use: ALLOW_FAILED_ACQUISITION=1 sbatch slurm/submit_audit_array.slurm")
        print(f"     • Or add: --allow-failed-acquisition to stage 02")
        
    else:
        print("\n✅ PASS: Adapter meets all quality thresholds!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
