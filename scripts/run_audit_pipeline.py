"""
Run audit pipeline stages for a resolved adapter configuration.

This wrapper orchestrates the audit pipeline stages, ensuring proper
sequencing and validation of artifacts. Can run individual stages or
the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run audit pipeline stages for a resolved adapter config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Resolved audit config (from resolve_audit_config.py)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Stages to run (default: 1-5 for discovery). Example: --stages 1 2 3",
    )
    parser.add_argument(
        "--skip-preregistration",
        action="store_true",
        help="Skip stage 00 (preregistration). Use when already registered.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use (default: python)",
    )
    return parser.parse_args()


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and validate audit config."""
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be JSON object: {config_path}")
    
    return config


def run_command(
    python_exe: str,
    script: str,
    args: list[str],
    cwd: Path | None = None,
) -> int:
    """Run a stage script and return exit code."""
    cmd = [python_exe, script] + args
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = read_config(args.config)
    project_root = Path.cwd()

    stage_map = {
        0: ("00_freeze_experiment.py", []),
        1: ("01_verify_adapter_acquisition.py", [
            "--phase", "generate", "--condition", "BASE",
        ]),
        2: ("02_generate_discovery_prompts.py", []),
        3: ("03_generate_discovery_rollouts.py", ["--condition", "BASE"]),
        4: ("04_run_open_diff_judge.py", []),
        5: ("05_cluster_hypotheses.py", []),
        6: ("06_generate_targeted_evals.py", []),
        7: ("07_generate_verification_rollouts.py", []),
        8: ("08_grade_verification.py", []),
        9: ("09_finalize_verified_labels.py", []),
        10: ("10_evaluate_meta_ia.py", []),
    }

    # Build common args
    common_args = ["--config", str(args.config)]

    print(f"Audit Pipeline")
    print(f"Config: {args.config}")
    print(f"Experiment: {config.get('experiment_name', '?')}")
    print(f"Output: {config.get('output_dir', '?')}")
    print(f"Stages to run: {args.stages}")

    failed_stages = []
    for stage_num in args.stages:
        if stage_num not in stage_map:
            print(f"ERROR: Unknown stage {stage_num}")
            sys.exit(1)

        script_name, stage_args = stage_map[stage_num]
        script_path = project_root / "scripts" / script_name

        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            sys.exit(1)

        invocations = [stage_args]
        if stage_num == 3:
            invocations = [
                ["--condition", "BASE"],
                ["--condition", "TARGET"],
            ]
        exit_code = 0
        for invocation_args in invocations:
            exit_code = run_command(
                args.python,
                str(script_path),
                common_args + invocation_args,
                cwd=project_root,
            )
            if exit_code != 0:
                break

        if exit_code != 0:
            print(f"\nERROR: Stage {stage_num} failed with exit code {exit_code}")
            failed_stages.append(stage_num)

    if failed_stages:
        print(f"\n\nPipeline failed at stages: {failed_stages}")
        sys.exit(1)

    print(f"\n\nPipeline completed successfully!")
    print(f"Output: {config.get('output_dir')}")


if __name__ == "__main__":
    main()
