#!/usr/bin/env python3
"""Return success only when a semantic pipeline gate permits downstream work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TERMINAL_EXIT_CODE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--gate", required=True, choices=("acquisition", "verified-labels")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def output_root(config_path: Path) -> Path:
    config = read_json(config_path)
    raw = config.get("output_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Config has no non-empty output_dir")
    expanded = raw.replace("${PROJECT}", str(PROJECT)).replace("$PROJECT", str(PROJECT))
    return Path(os.path.expandvars(os.path.expanduser(expanded))).resolve()


def main() -> int:
    args = parse_args()
    root = output_root(args.config)
    if args.gate == "acquisition":
        path = root / "acquisition/gate.json"
        status = read_json(path).get("status")
        if status == "PASS":
            print(f"PASS acquisition gate: {path}")
            return 0
        if status in {"FAIL", "FAILED"}:
            print(f"TERMINAL acquisition gate status={status}: {path}")
            return TERMINAL_EXIT_CODE
        raise ValueError(f"Unknown acquisition gate status {status!r}: {path}")

    path = root / "verification/finalization_status_v1.json"
    payload = read_json(path)
    status = payload.get("status")
    if status == "verified_labels_frozen":
        labels = root / "verified_labels/labels_v1.jsonl"
        if not labels.is_file() or labels.stat().st_size == 0:
            raise ValueError(f"Stage 09 claims verified labels but file is missing: {labels}")
        print(f"PASS verified-label gate: {payload.get('num_verified_labels')} labels")
        return 0
    if status == "no_verified_labels":
        print(f"TERMINAL audit outcome: no verified labels ({path})")
        return TERMINAL_EXIT_CODE
    raise ValueError(f"Unknown Stage-09 finalization status {status!r}: {path}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
