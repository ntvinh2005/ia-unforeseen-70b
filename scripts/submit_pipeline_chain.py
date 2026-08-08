#!/usr/bin/env python3
"""Submit the remaining audit as separate Slurm jobs connected by dependencies."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from submit_stage_batch import (
    PROJECT,
    STAGES,
    StageSpec,
    adapter_name,
    all_nonempty,
    load_config_paths,
    output_root,
    read_config,
    safe_job_name,
)


@dataclass(frozen=True)
class ChainStep:
    name: str
    spec: StageSpec
    gate: str | None = None


ACQUISITION_GUARD = StageSpec(
    "pipeline_gate.slurm",
    ("acquisition/gate.json",),
    (),
    (("PIPELINE_GATE", "acquisition"),),
)
LABEL_GUARD = StageSpec(
    "pipeline_gate.slurm",
    ("verification/finalization_status_v1.json",),
    (),
    (("PIPELINE_GATE", "verified-labels"),),
)

CHAIN: tuple[ChainStep, ...] = (
    ChainStep("00-freeze", STAGES["00-freeze"]),
    ChainStep("01-base", STAGES["01-base"]),
    ChainStep("01-target", STAGES["01-target"]),
    ChainStep("01-grade", STAGES["01-grade"]),
    ChainStep("01-summarize", STAGES["01-summarize"]),
    ChainStep("gate-acquisition", ACQUISITION_GUARD, "acquisition"),
    ChainStep("02-prompts", STAGES["02-prompts"]),
    ChainStep("03-base", STAGES["03-base"]),
    ChainStep("03-target", STAGES["03-target"]),
    ChainStep("04-judge", STAGES["04-judge"]),
    ChainStep("05-cluster", STAGES["05-cluster"]),
    ChainStep("06-targeted-auto", STAGES["06-targeted-auto"]),
    ChainStep("07-dev-base", STAGES["07-dev-base"]),
    ChainStep("07-dev-target", STAGES["07-dev-target"]),
    ChainStep("08-dev-grade", STAGES["08-dev-grade"]),
    ChainStep("08-dev-summarize", STAGES["08-dev-summarize"]),
    ChainStep("07-test-base", STAGES["07-test-base"]),
    ChainStep("07-test-target", STAGES["07-test-target"]),
    ChainStep("08-test-grade", STAGES["08-test-grade"]),
    ChainStep("08-test-summarize", STAGES["08-test-summarize"]),
    ChainStep("09-finalize-auto", STAGES["09-finalize-auto"]),
    ChainStep("gate-verified-labels", LABEL_GUARD, "verified-labels"),
    ChainStep("10-target", STAGES["10-target"]),
    ChainStep("10-base-ia", STAGES["10-base-ia"]),
    ChainStep("10-target-ia", STAGES["10-target-ia"]),
    ChainStep("10-grade", STAGES["10-grade"]),
    ChainStep("10-summarize", STAGES["10-summarize"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit all remaining automatic checkpoints as individual afterok jobs. "
            "Dry-run is the default."
        )
    )
    parser.add_argument("--configs", nargs="*", type=Path, default=())
    parser.add_argument("--config-list", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--stop-after",
        choices=("verified-labels", "full"),
        default="full",
        help=(
            "Stop after the verified-label gate (Stage 09) or include Stage 10 "
            "Meta-IA evaluation. Default: full."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the dependency chains",
    )
    return parser.parse_args()


def active_jobs() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    jobs: dict[str, str] = {}
    duplicates: set[str] = set()
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, name = line.strip().split("|", 1)
        if name in jobs:
            duplicates.add(name)
        jobs[name] = job_id
    for name in duplicates:
        jobs.pop(name, None)
    return jobs


def existing_gate_outcome(root: Path, gate: str) -> str | None:
    if gate == "acquisition":
        path = root / "acquisition/gate.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        status = value.get("status") if isinstance(value, dict) else None
        if status == "PASS":
            return "pass"
        if status in {"FAIL", "FAILED"}:
            return "terminal"
        return "error"
    path = root / "verification/finalization_status_v1.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    status = value.get("status") if isinstance(value, dict) else None
    if status == "verified_labels_frozen":
        return "pass"
    if status == "no_verified_labels":
        return "terminal"
    return "error"


def submit_command(
    *,
    step: ChainStep,
    adapter: str,
    audit_config: Path,
    dependency: str | None,
) -> tuple[str, list[str]]:
    job_name = safe_job_name(step.name, adapter)
    exports = {"AUDIT_CONFIG": str(audit_config), **dict(step.spec.environment)}
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--export",
        "ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
    ]
    if dependency is not None:
        command.extend(
            [
                "--dependency",
                f"afterok:{dependency}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    command.append(str(PROJECT / "slurm/stages" / step.spec.script))
    return job_name, command


def main() -> int:
    args = parse_args()
    configs = load_config_paths(args)
    active = active_jobs() if args.submit else {}
    chain = CHAIN
    if args.stop_after == "verified-labels":
        stop = next(
            index
            for index, step in enumerate(CHAIN)
            if step.name == "gate-verified-labels"
        )
        chain = CHAIN[: stop + 1]
    records: list[dict[str, object]] = []
    errors = 0

    for resolved_config in configs:
        config = read_config(resolved_config)
        root = output_root(config)
        adapter = adapter_name(config, root)
        frozen_config = root / "config.yaml"
        available = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.stat().st_size > 0
        } if root.is_dir() else set()
        dependency: str | None = None
        terminal = False
        print(f"\n=== {adapter} ===")

        for position, step in enumerate(chain, start=1):
            base_record: dict[str, object] = {
                "adapter": adapter,
                "output_root": str(root),
                "position": position,
                "stage": step.name,
            }
            if terminal:
                base_record.update(action="not_applicable", reason="earlier terminal gate")
                records.append(base_record)
                continue

            if step.gate is not None:
                outcome = existing_gate_outcome(root, step.gate)
                if outcome == "pass":
                    if step.gate == "verified-labels":
                        available.add("verified_labels/labels_v1.jsonl")
                    base_record.update(action="skip", reason="gate already passed")
                    records.append(base_record)
                    print(f"SKIP       {step.name}: gate already passed")
                    continue
                if outcome == "terminal":
                    base_record.update(action="terminal", reason=f"{step.gate} terminal")
                    records.append(base_record)
                    terminal = True
                    print(f"TERMINAL   {step.name}: downstream stages will not run")
                    continue
                if outcome == "error":
                    base_record.update(action="error", reason="malformed gate status")
                    records.append(base_record)
                    errors += 1
                    terminal = True
                    print(f"ERROR      {step.name}: malformed gate status")
                    continue

            if step.spec.completed and all_nonempty(root, step.spec.completed):
                available.update(step.spec.completed)
                base_record.update(action="skip", reason="already complete")
                records.append(base_record)
                print(f"SKIP       {step.name}: already complete")
                continue

            missing = [item for item in step.spec.required if item not in available]
            if missing:
                base_record.update(action="error", reason=f"unproduced inputs: {missing}")
                records.append(base_record)
                errors += 1
                terminal = True
                print(f"ERROR      {step.name}: unproduced inputs: {', '.join(missing)}")
                continue

            audit_config = resolved_config if step.name == "00-freeze" else frozen_config
            # The frozen config may be produced by a scheduled Stage 00 job.
            if step.name != "00-freeze" and "config.yaml" not in available:
                base_record.update(action="error", reason="config.yaml is not produced")
                records.append(base_record)
                errors += 1
                terminal = True
                print(f"ERROR      {step.name}: frozen config is unavailable")
                continue

            job_name, command = submit_command(
                step=step,
                adapter=adapter,
                audit_config=audit_config,
                dependency=dependency,
            )
            base_record.update(job_name=job_name, dependency=dependency, command=command)
            active_id = active.get(job_name)
            if active_id is not None:
                dependency = active_id
                base_record.update(action="active", job_id=active_id)
                print(f"ACTIVE     {step.name}: {active_id}")
            elif args.submit:
                try:
                    result = subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    base_record.update(
                        action="error",
                        reason=(exc.stderr or exc.stdout or str(exc)).strip(),
                    )
                    records.append(base_record)
                    errors += 1
                    terminal = True
                    print(f"ERROR      {step.name}: submission failed")
                    continue
                dependency = result.stdout.strip().split(";", 1)[0]
                base_record.update(action="submitted", job_id=dependency)
                active[job_name] = dependency
                print(f"SUBMITTED  {step.name}: {dependency}")
            else:
                dependency = f"DRY{position:02d}"
                base_record.update(action="would_submit", dry_job_id=dependency)
                print(
                    f"WOULD_SUBMIT {step.name}: "
                    f"afterok={base_record.get('dependency') or 'none'}"
                )

            available.update(step.spec.completed)
            if step.gate == "verified-labels":
                # The guard succeeds only when Stage 09 produced this file. If
                # there are no labels, it exits nonzero and afterok cancels all
                # Stage 10 jobs before they allocate GPUs.
                available.add("verified_labels/labels_v1.jsonl")
            records.append(base_record)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.submit:
        directory = PROJECT / "logs/pipeline_chains"
        directory.mkdir(parents=True, exist_ok=True)
        receipt_stem = (
            safe_job_name("chain", adapter_name(read_config(configs[0]), output_root(read_config(configs[0]))))
            if len(configs) == 1
            else f"chain_{len(configs)}configs"
        )
        receipt = directory / f"{timestamp}_{receipt_stem}_{os.getpid()}_pipeline_chain.jsonl"
        receipt.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        print(f"\nChain receipt: {receipt}")
    else:
        print("\nDry-run only. Re-run with --submit after reviewing the chain.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
