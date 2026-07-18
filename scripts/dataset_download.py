"""Download the project datasets into a configurable, pinned local tree."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


DATASETS = (
    "llama-harmful-mo-training-data",
    "rare-mo-eval-data",
    "quirk-mo-eval-data",
    "problematic-mo-eval-data",
    "prism4-mo-eval-data",
    "heuristic-mo-eval-data",
    "harmful-benign-mo-eval-data",
    "backdoor-mo-eval-data",
    "sandbagging-mo-eval-data",
    "encrypted-harm-mo-eval-data",
    "ukaisi-sandbaggers-mo-eval-data",
)


def project_root() -> Path:
    configured = os.environ.get("PROJECT")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and pin the Hugging Face datasets used by this project."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "datasets",
        help="Destination directory (default: $PROJECT/datasets).",
    )
    parser.add_argument(
        "--organization",
        default="introspection-auditing",
        help="Hugging Face dataset organization.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Download only this dataset name; may be passed repeatedly.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision applied to every selected dataset; latest commits are pinned otherwise.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Download manifest path (default: OUTPUT_ROOT/download_manifest.json).",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    selected = tuple(args.datasets or DATASETS)
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {unknown}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / "download_manifest.json"
    )
    api = HfApi()
    records: list[dict[str, Any]] = []

    for index, dataset_name in enumerate(selected, start=1):
        repo_id = f"{args.organization}/{dataset_name}"
        destination = output_root / dataset_name
        record: dict[str, Any] = {
            "dataset": dataset_name,
            "repo_id": repo_id,
            "destination": str(destination),
            "status": "started",
        }
        print(f"[{index}/{len(selected)}] {repo_id} -> {destination}", flush=True)
        try:
            revision = args.revision or api.dataset_info(repo_id=repo_id).sha
            if not revision:
                raise RuntimeError(f"Could not resolve an immutable revision for {repo_id}")
            record["revision"] = revision
            record["snapshot_path"] = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=str(destination),
            )
            record["status"] = "success"
        except Exception as exc:  # continue so one outage does not hide later failures
            record.update(
                {"status": "failure", "error_type": type(exc).__name__, "error": str(exc)}
            )
            print(f"Failed: {repo_id}: {exc}", flush=True)
        records.append(record)
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "output_root": str(output_root),
                "datasets": records,
            },
        )

    failures = [record for record in records if record["status"] != "success"]
    print(
        f"Downloaded {len(records) - len(failures)}/{len(records)} datasets; "
        f"manifest: {manifest_path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
