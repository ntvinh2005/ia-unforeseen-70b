#!/usr/bin/env python3
"""
Helper script to count ready adapters and submit SLURM array job.

Usage:
    python scripts/submit_audit_array_job.py [--submit]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count ready adapters and optionally submit SLURM array job."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(os.path.expandvars("$PROJECT/configs/adapter_registry.json")),
        help="Path to adapter registry.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the SLURM job (dry-run mode otherwise).",
    )
    parser.add_argument(
        "--slurm-script",
        type=Path,
        default=Path(os.path.expandvars("$PROJECT/slurm/submit_audit_array.slurm")),
        help="Path to SLURM submission script.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(os.path.expandvars("$PROJECT")),
        help="Project root directory.",
    )
    return parser.parse_args()


def count_ready_adapters(registry_path: Path) -> int:
    """Count the number of ready adapters in the registry."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, dict):
        raise ValueError(f"Invalid registry: {registry_path}")

    adapters = registry.get("adapters", [])
    ready_count = sum(
        1 for adapter in adapters if isinstance(adapter, dict) and adapter.get("status") == "ready"
    )

    return ready_count


def print_adapter_summary(registry_path: Path) -> None:
    """Print summary of adapters in the registry."""
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    adapters = registry.get("adapters", [])
    total = len(adapters)
    ready = [a for a in adapters if isinstance(a, dict) and a.get("status") == "ready"]
    unresolved = [a for a in adapters if isinstance(a, dict) and a.get("status") != "ready"]

    print(f"\nAdapter Registry Summary:")
    print(f"  Registry: {registry_path}")
    print(f"  Generated: {registry.get('generated_at', 'unknown')}")
    print(f"  Total adapters: {total}")
    print(f"  Ready: {len(ready)}")
    print(f"  Unresolved: {len(unresolved)}")

    if ready:
        print(f"\nReady adapters (0-{len(ready) - 1}):")
        for i, adapter in enumerate(ready):
            name = adapter.get("name", "unknown")
            behavior = adapter.get("intended_behavior", "unknown")[:60]
            print(f"  [{i}] {name}")
            print(f"      → {behavior}...")

    if unresolved:
        print(f"\nUnresolved adapters:")
        for adapter in unresolved:
            name = adapter.get("name", "unknown")
            status = adapter.get("status", "unknown")
            missing = adapter.get("missing_fields", [])
            print(f"  [{status}] {name}")
            if missing:
                print(f"      Missing: {', '.join(missing)}")


def submit_slurm_array(
    slurm_script: Path,
    num_adapters: int,
    project: Path,
    dry_run: bool = True,
) -> None:
    """Submit SLURM array job."""
    if num_adapters <= 0:
        print("ERROR: No ready adapters found. Cannot submit job.", file=sys.stderr)
        sys.exit(1)

    array_arg = f"--array=0-{num_adapters - 1}"

    cmd = [
        "sbatch",
        array_arg,
        str(slurm_script),
    ]

    print(f"\nSLURM Command:")
    print(f"  {' '.join(cmd)}")
    print(f"\nThis will submit {num_adapters} parallel audit jobs.")
    print(f"Array task IDs: 0 to {num_adapters - 1}")

    if dry_run:
        print("\n[DRY RUN] Use --submit to actually submit the job.")
        return

    env = os.environ.copy()
    env["PROJECT"] = str(project)

    print("\nSubmitting...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(result.stdout)
    print(f"\nCheck job status: squeue -u $(whoami) -j <job_id>")
    print(f"View logs: tail -f {project}/logs/audit_<job_id>_*.err")


def main() -> None:
    args = parse_args()

    # Expand environment variables
    registry_path = Path(os.path.expandvars(str(args.registry)))
    slurm_script = Path(os.path.expandvars(str(args.slurm_script)))
    project = Path(os.path.expandvars(str(args.project)))

    try:
        num_ready = count_ready_adapters(registry_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print_adapter_summary(registry_path)

    if num_ready == 0:
        print("\nNo ready adapters. Run register_adapters.py first.", file=sys.stderr)
        sys.exit(1)

    submit_slurm_array(
        slurm_script=slurm_script,
        num_adapters=num_ready,
        project=project,
        dry_run=not args.submit,
    )


if __name__ == "__main__":
    main()
