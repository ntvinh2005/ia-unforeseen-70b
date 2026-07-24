#!/usr/bin/env python3
"""Inspect or intercept a pipeline dependency-chain receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Chain JSONL receipt; defaults to the newest receipt",
    )
    parser.add_argument("--adapter", help="Limit display/cancellation to one adapter")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--cancel-pending",
        action="store_true",
        help="Cancel only jobs that have not started",
    )
    action.add_argument(
        "--cancel-all",
        action="store_true",
        help="Cancel pending and currently running jobs",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation for cancellation",
    )
    return parser.parse_args()


def receipt_path(value: Path | None) -> Path:
    if value is not None:
        path = value.resolve()
    else:
        candidates = sorted(
            (PROJECT / "logs/pipeline_chains").glob("*_pipeline_chain.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError("No pipeline-chain receipt exists")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_records(path: Path, adapter: str | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Receipt line {line_number} is not an object")
        if adapter is None or value.get("adapter") == adapter:
            records.append(value)
    if not records:
        raise ValueError("No receipt records match the requested adapter")
    return records


def scheduler_states(job_ids: list[str]) -> tuple[dict[str, tuple[str, str, str]], dict[str, str]]:
    accounting: dict[str, tuple[str, str, str]] = {}
    queue_reasons: dict[str, str] = {}
    if not job_ids:
        return accounting, queue_reasons
    joined = ",".join(job_ids)
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            joined,
            "--format=JobIDRaw,State,Elapsed,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] in job_ids:
            accounting[fields[0]] = (fields[1], fields[2], fields[3])
    queued = subprocess.run(
        ["squeue", "-h", "-j", joined, "-o", "%i|%R"],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in queued.stdout.splitlines():
        if "|" in line:
            job_id, reason = line.split("|", 1)
            queue_reasons[job_id.strip()] = reason.strip()
    return accounting, queue_reasons


def main() -> int:
    args = parse_args()
    path = receipt_path(args.receipt)
    records = load_records(path, args.adapter)
    job_ids = [
        str(record["job_id"])
        for record in records
        if record.get("action") in {"submitted", "active"} and record.get("job_id")
    ]
    accounting, reasons = scheduler_states(job_ids)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("adapter", "unknown"))].append(record)

    print(f"Receipt: {path}")
    cancellable_pending: list[str] = []
    cancellable_all: list[str] = []
    for adapter, adapter_records in grouped.items():
        print(f"\n=== {adapter} ===")
        running: list[str] = []
        pending: list[tuple[str, str]] = []
        for record in adapter_records:
            stage = str(record.get("stage", "?"))
            action = str(record.get("action", "?"))
            job_id = str(record.get("job_id", ""))
            if job_id:
                state, elapsed, exit_code = accounting.get(
                    job_id, ("UNKNOWN", "-", "-")
                )
                reason = reasons.get(job_id, "")
                suffix = f" reason={reason}" if reason else ""
                print(
                    f"{stage:24} job={job_id:10} state={state:20} "
                    f"elapsed={elapsed:10} exit={exit_code}{suffix}"
                )
                normalized = state.split("+", 1)[0]
                if normalized in ACTIVE_STATES:
                    cancellable_all.append(job_id)
                if normalized in {"RUNNING", "CONFIGURING", "COMPLETING"}:
                    running.append(f"{stage}:{normalized}")
                if normalized == "PENDING":
                    cancellable_pending.append(job_id)
                    pending.append((stage, reason or "PENDING"))
            elif action not in {"not_applicable"}:
                print(f"{stage:24} {action}: {record.get('reason', '')}")
        if running:
            print("Current:", ", ".join(running))
        elif pending:
            print(f"Current: waiting for {pending[0][0]} ({pending[0][1]})")
        else:
            print("Current: no active jobs")
        print(f"Remaining queued checkpoints: {len(pending)}")
        root_raw = adapter_records[0].get("output_root")
        if isinstance(root_raw, str):
            status_path = (
                Path(root_raw) / "verification/finalization_status_v1.json"
            )
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                print(
                    "Finalization:",
                    status.get("status"),
                    f"labels={status.get('num_verified_labels')}",
                )

    if args.cancel_pending or args.cancel_all:
        if not args.yes:
            raise SystemExit("Cancellation requires --yes")
        targets = cancellable_pending if args.cancel_pending else cancellable_all
        targets = list(dict.fromkeys(targets))
        if not targets:
            print("\nNo matching active jobs to cancel.")
            return 0
        subprocess.run(["scancel", *targets], check=True)
        print(f"\nCancelled {len(targets)} jobs: {', '.join(targets)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
