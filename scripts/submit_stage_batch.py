#!/usr/bin/env python3
"""Safely submit one audit checkpoint for a bounded adapter batch.

This launcher never chains audit stages.  It validates durable prerequisites,
skips completed/terminal adapters, and submits at most one Slurm job per
selected adapter.  Dry-run is the default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StageSpec:
    script: str
    required: tuple[str, ...]
    completed: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()
    needs_verified_labels: bool = False
    requires_passed_acquisition: bool = False


STAGES: dict[str, StageSpec] = {
    "00-freeze": StageSpec(
        "00_freeze.slurm", (), ("preregistration_manifest.json", "config.yaml")
    ),
    "01-base": StageSpec(
        "01_acquisition.slurm",
        ("preregistration_manifest.json",),
        ("acquisition/base_rollouts.jsonl",),
        (("PHASE", "generate"), ("CONDITION", "BASE")),
    ),
    "01-target": StageSpec(
        "01_acquisition.slurm",
        ("preregistration_manifest.json",),
        ("acquisition/target_rollouts.jsonl",),
        (("PHASE", "generate"), ("CONDITION", "TARGET")),
    ),
    "01-grade": StageSpec(
        "01_acquisition.slurm",
        ("acquisition/base_rollouts.jsonl", "acquisition/target_rollouts.jsonl"),
        ("acquisition/judgments.jsonl",),
        (("PHASE", "grade"),),
    ),
    "01-summarize": StageSpec(
        "01_acquisition_summarize.slurm",
        ("acquisition/judgments.jsonl",),
        ("acquisition/gate.json",),
    ),
    "02-prompts": StageSpec(
        "02_prompts.slurm",
        ("acquisition/gate.json",),
        ("frozen_manifest.json", "prompts/discovery.jsonl"),
        requires_passed_acquisition=True,
    ),
    "03-base": StageSpec(
        "03_rollouts.slurm",
        ("frozen_manifest.json", "prompts/discovery.jsonl"),
        ("rollouts/base/discovery.jsonl",),
        (("CONDITION", "BASE"),),
    ),
    "03-target": StageSpec(
        "03_rollouts.slurm",
        ("frozen_manifest.json", "prompts/discovery.jsonl"),
        ("rollouts/target/discovery.jsonl",),
        (("CONDITION", "TARGET"),),
    ),
    "04-judge": StageSpec(
        "04_judge.slurm",
        ("rollouts/base/discovery.jsonl", "rollouts/target/discovery.jsonl"),
        ("discovery_judgments/judgments.jsonl",),
    ),
    "05-cluster": StageSpec(
        "05_cluster.slurm",
        ("discovery_judgments/judgments.jsonl",),
        ("hypotheses/clustered_candidates.json",),
    ),
    "06-targeted": StageSpec(
        "06_targeted_prompts.slurm",
        ("hypotheses/clustered_candidates.json", "hypotheses/human_reviewed.json"),
        ("prompts/targeted_dev.jsonl", "prompts/targeted_test.jsonl"),
    ),
    "06-targeted-auto": StageSpec(
        "06_targeted_prompts.slurm",
        ("hypotheses/clustered_candidates.json",),
        ("prompts/targeted_dev.jsonl", "prompts/targeted_test.jsonl"),
        (("APPROVE_ALL", "1"),),
    ),
    "07-dev-base": StageSpec(
        "07_verification_rollouts.slurm",
        ("prompts/targeted_dev.jsonl",),
        ("rollouts/base/verification_dev.jsonl",),
        (("SPLIT", "dev"), ("CONDITION", "BASE")),
    ),
    "07-dev-target": StageSpec(
        "07_verification_rollouts.slurm",
        ("prompts/targeted_dev.jsonl",),
        ("rollouts/target/verification_dev.jsonl",),
        (("SPLIT", "dev"), ("CONDITION", "TARGET")),
    ),
    "08-dev-grade": StageSpec(
        "08_verification_grade.slurm",
        (
            "rollouts/base/verification_dev.jsonl",
            "rollouts/target/verification_dev.jsonl",
        ),
        ("verification/dev_judgments.jsonl",),
        (("SPLIT", "dev"), ("PHASE", "grade")),
    ),
    "08-dev-summarize": StageSpec(
        "08_verification_summarize.slurm",
        ("verification/dev_judgments.jsonl",),
        ("verification/dev_metrics.json", "verification/dev_bootstrap_results.json"),
        (("SPLIT", "dev"),),
    ),
    "07-test-base": StageSpec(
        "07_verification_rollouts.slurm",
        ("verification/dev_metrics.json", "prompts/targeted_test.jsonl"),
        ("rollouts/base/verification_test.jsonl",),
        (("SPLIT", "test"), ("CONDITION", "BASE")),
    ),
    "07-test-target": StageSpec(
        "07_verification_rollouts.slurm",
        ("verification/dev_metrics.json", "prompts/targeted_test.jsonl"),
        ("rollouts/target/verification_test.jsonl",),
        (("SPLIT", "test"), ("CONDITION", "TARGET")),
    ),
    "08-test-grade": StageSpec(
        "08_verification_grade.slurm",
        (
            "rollouts/base/verification_test.jsonl",
            "rollouts/target/verification_test.jsonl",
        ),
        ("verification/test_judgments.jsonl",),
        (("SPLIT", "test"), ("PHASE", "grade")),
    ),
    "08-test-summarize": StageSpec(
        "08_verification_summarize.slurm",
        ("verification/test_judgments.jsonl",),
        ("verification/test_metrics.json", "verification/test_bootstrap_results.json"),
        (("SPLIT", "test"),),
    ),
    "09-finalize": StageSpec(
        "09_finalize.slurm",
        (
            "verification/test_judgments.jsonl",
            "verification/test_metrics.json",
            "verification/human_label_reviews.json",
            "verification/calibration.jsonl",
        ),
        ("verification/finalization_status_v1.json",),
    ),
    "09-finalize-auto": StageSpec(
        "09_finalize.slurm",
        ("verification/test_judgments.jsonl", "verification/test_metrics.json"),
        ("verification/finalization_status_v1.json",),
        (("AUTO_APPROVE_ALL", "1"),),
    ),
    "10-target": StageSpec(
        "10_meta_ia.slurm",
        ("verified_labels/labels_v1.jsonl",),
        ("meta_ia_evaluation/rollouts_target.jsonl",),
        (("PHASE", "rollouts"), ("CONDITION", "TARGET")),
        needs_verified_labels=True,
    ),
    "10-base-ia": StageSpec(
        "10_meta_ia.slurm",
        ("verified_labels/labels_v1.jsonl",),
        ("meta_ia_evaluation/rollouts_base_ia.jsonl",),
        (("PHASE", "rollouts"), ("CONDITION", "BASE_IA")),
        needs_verified_labels=True,
    ),
    "10-target-ia": StageSpec(
        "10_meta_ia.slurm",
        ("verified_labels/labels_v1.jsonl",),
        ("meta_ia_evaluation/rollouts_target_ia.jsonl",),
        (("PHASE", "rollouts"), ("CONDITION", "TARGET_IA")),
        needs_verified_labels=True,
    ),
    "10-grade": StageSpec(
        "10_meta_ia.slurm",
        (
            "meta_ia_evaluation/rollouts_target.jsonl",
            "meta_ia_evaluation/rollouts_base_ia.jsonl",
            "meta_ia_evaluation/rollouts_target_ia.jsonl",
        ),
        ("meta_ia_evaluation/judgments.jsonl",),
        (("PHASE", "grade"),),
        needs_verified_labels=True,
    ),
    "10-summarize": StageSpec(
        "10_meta_ia_summarize.slurm",
        ("meta_ia_evaluation/judgments.jsonl",),
        ("meta_ia_evaluation/metrics.json",),
        needs_verified_labels=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one checkpoint for a bounded adapter batch. Dry-run by default; "
            "run the command again with --submit after reviewing its decisions."
        )
    )
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument(
        "--configs",
        nargs="*",
        type=Path,
        default=(),
        help="Persistent resolved config files; shell globs are allowed",
    )
    parser.add_argument(
        "--config-list",
        type=Path,
        help="Text file containing one persistent resolved config path per line",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum adapters selected in this invocation (default: 5)",
    )
    parser.add_argument(
        "--allow-failed-acquisition",
        action="store_true",
        help="Allow Stage 02 to quarantine and continue a failed acquisition gate",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually call sbatch; without this flag the launcher only prints a dry-run",
    )
    return parser.parse_args()


def load_config_paths(args: argparse.Namespace) -> list[Path]:
    values = list(args.configs)
    if args.config_list is not None:
        for raw in args.config_list.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if value and not value.startswith("#"):
                values.append(Path(value))
    unique: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise SystemExit("No configs supplied; use --configs or --config-list")
    if args.offset < 0 or args.limit < 1:
        raise SystemExit("--offset must be non-negative and --limit must be positive")
    return unique[args.offset : args.offset + args.limit]


def read_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config is not a JSON object: {path}")
    return value


def output_root(config: dict[str, object]) -> Path:
    raw = config.get("output_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("config has no non-empty output_dir")
    # Resolved configs intentionally retain $PROJECT for portability. The
    # launcher knows its own repository root and must not depend on callers
    # remembering to export PROJECT in every login shell.
    expanded = raw.replace("${PROJECT}", str(PROJECT)).replace("$PROJECT", str(PROJECT))
    return Path(os.path.expandvars(os.path.expanduser(expanded))).resolve()


def adapter_name(config: dict[str, object], root: Path) -> str:
    adapter = config.get("behavior_adapter")
    if isinstance(adapter, dict):
        name = adapter.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return root.name


def all_nonempty(root: Path, relative_paths: Iterable[str]) -> bool:
    return all(
        (root / item).is_file() and (root / item).stat().st_size > 0
        for item in relative_paths
    )


def missing_inputs(root: Path, relative_paths: Iterable[str]) -> list[str]:
    return [
        item
        for item in relative_paths
        if not (root / item).is_file() or (root / item).stat().st_size == 0
    ]


def finalization_status(root: Path) -> str | None:
    path = root / "verification/finalization_status_v1.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    status = value.get("status") if isinstance(value, dict) else None
    return status if isinstance(status, str) else None


def active_job_names() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def safe_job_name(stage: str, adapter: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "-", adapter).strip("-")[:55]
    prefix = stage.replace("-summarize", "-sum")
    return f"audit-{prefix}-{suffix}"[:100]


def main() -> int:
    args = parse_args()
    spec = STAGES[args.stage]
    configs = load_config_paths(args)
    active = active_job_names() if args.submit else set()
    records: list[dict[str, object]] = []

    for resolved_config in configs:
        record: dict[str, object] = {
            "stage": args.stage,
            "resolved_config": str(resolved_config),
        }
        try:
            config = read_config(resolved_config)
            root = output_root(config)
            adapter = adapter_name(config, root)
            frozen_config = root / "config.yaml"
            job_name = safe_job_name(args.stage, adapter)
            record.update(adapter=adapter, output_root=str(root), job_name=job_name)

            if all_nonempty(root, spec.completed):
                record.update(action="skip", reason="already complete")
            elif job_name in active:
                record.update(action="skip", reason="matching job is already queued/running")
            elif spec.needs_verified_labels and finalization_status(root) == "no_verified_labels":
                record.update(action="terminal", reason="Stage 09 found no verified labels")
            else:
                missing = missing_inputs(root, spec.required)
                if missing:
                    record.update(action="blocked", reason=f"missing: {', '.join(missing)}")
                elif (
                    spec.requires_passed_acquisition
                    and not args.allow_failed_acquisition
                    and json.loads((root / "acquisition/gate.json").read_text()).get("status")
                    != "PASS"
                ):
                    record.update(
                        action="terminal",
                        reason=(
                            "acquisition gate failed "
                            "(use --allow-failed-acquisition to quarantine)"
                        ),
                    )
                else:
                    audit_config = resolved_config if args.stage == "00-freeze" else frozen_config
                    if not audit_config.is_file():
                        record.update(
                            action="blocked",
                            reason=f"missing persistent AUDIT_CONFIG: {audit_config}",
                        )
                    else:
                        environment = dict(spec.environment)
                        if args.stage == "02-prompts" and args.allow_failed_acquisition:
                            environment["ALLOW_FAILED_ACQUISITION"] = "1"
                        exports = {
                            "AUDIT_CONFIG": str(audit_config),
                            **environment,
                        }
                        export_arg = "ALL," + ",".join(
                            f"{key}={value}" for key, value in exports.items()
                        )
                        command = [
                            "sbatch",
                            "--parsable",
                            "--job-name",
                            job_name,
                            "--export",
                            export_arg,
                            str(PROJECT / "slurm/stages" / spec.script),
                        ]
                        if args.submit:
                            result = subprocess.run(
                                command, check=True, capture_output=True, text=True
                            )
                            record.update(
                                action="submitted",
                                job_id=result.stdout.strip().split(";", 1)[0],
                                command=command,
                            )
                            active.add(job_name)
                        else:
                            record.update(action="would_submit", command=command)
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as exc:
            record.update(action="error", reason=str(exc))
        records.append(record)
        print(
            f"{str(record.get('action', 'error')).upper():12} "
            f"{record.get('adapter', resolved_config.stem)}: "
            f"{record.get('reason', record.get('job_id', 'ready'))}"
        )
        if record.get("action") == "would_submit":
            print("  " + " ".join(str(item) for item in record["command"]))

    counts: dict[str, int] = {}
    for record in records:
        action = str(record.get("action", "error"))
        counts[action] = counts.get(action, 0) + 1
    print("Summary:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))

    if args.submit:
        directory = PROJECT / "logs/batch_submissions"
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        receipt = directory / f"{timestamp}_{args.stage}.jsonl"
        receipt.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
        print(f"Submission receipt: {receipt}")
    return 1 if any(item.get("action") == "error" for item in records) else 0


if __name__ == "__main__":
    sys.exit(main())
