#!/usr/bin/env python3
"""Resolve a bounded slice of ready adapters into persistent config files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare persistent per-adapter configs in small registry slices. "
            "Preview by default; pass --resolve to download/resolve."
        )
    )
    parser.add_argument(
        "--registry", type=Path, default=PROJECT / "configs/adapter_registry.json"
    )
    parser.add_argument(
        "--base-config", type=Path, default=PROJECT / "configs/audit_base.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "configs/resolved"
    )
    parser.add_argument(
        "--adapter-cache", type=Path, default=PROJECT / ".cache/hf_adapters"
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=PROJECT / "envs/audit_env/bin/python3",
        help="Python executable used by the resolver",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Actually invoke resolve_audit_config.py; otherwise only preview",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offset < 0 or args.limit < 1:
        raise SystemExit("--offset must be non-negative and --limit must be positive")
    registry_path = args.registry.resolve()
    base_config = args.base_config.resolve()
    output_dir = args.output_dir.resolve()
    adapter_cache = args.adapter_cache.resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    adapters = payload.get("adapters")
    if not isinstance(adapters, list):
        raise SystemExit("registry.adapters must be an array")
    ready = [
        item
        for item in adapters
        if isinstance(item, dict)
        and item.get("status") == "ready"
        and isinstance(item.get("name"), str)
    ]
    selected = ready[args.offset : args.offset + args.limit]
    if not selected:
        raise SystemExit("Selected registry slice is empty")

    # Preserve the virtual-environment symlink. Path.resolve() would replace it
    # with the base interpreter and lose the venv's site-packages.
    python = args.python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit(f"Python is not executable: {python}")
    paths: list[Path] = []
    failures = 0
    for item in selected:
        name = str(item["name"])
        target = output_dir / f"{name}.json"
        paths.append(target)
        if target.is_file() and target.stat().st_size > 0:
            print(f"READY   {name}: {target}")
            continue
        command = [
            str(python),
            str(PROJECT / "scripts/resolve_audit_config.py"),
            "--base-config",
            str(base_config),
            "--registry",
            str(registry_path),
            "--adapter-name",
            name,
            "--hf-adapter-cache",
            str(adapter_cache),
            "--output",
            str(target),
        ]
        if not args.resolve:
            print(f"WOULD_RESOLVE {name}")
            print("  " + " ".join(command))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, check=True)
            print(f"RESOLVED {name}: {target}")
        except subprocess.CalledProcessError as exc:
            failures += 1
            print(f"ERROR    {name}: resolver exited {exc.returncode}", file=sys.stderr)

    if args.resolve:
        usable = [path for path in paths if path.is_file() and path.stat().st_size > 0]
        list_path = output_dir / (
            f"batch_{args.offset:03d}_{args.offset + len(selected) - 1:03d}.txt"
        )
        list_path.write_text(
            "".join(str(path.resolve()) + "\n" for path in usable),
            encoding="utf-8",
        )
        print(f"Config list ({len(usable)} adapters): {list_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
