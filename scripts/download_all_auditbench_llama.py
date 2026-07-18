from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


ORG = "auditing-agents"

# Matches the six core configurations:
#
# transcripts_only
# transcripts_only_then_redteam_high
# transcripts_only_then_redteam_kto
# synth_docs_only
# synth_docs_only_then_redteam_high
# synth_docs_only_then_redteam_kto
CORE_PATTERN = re.compile(
    r"^auditing-agents/llama_70b_"
    r"(transcripts_only|synth_docs_only)"
    r"(?:_then_redteam_(high|kto))?"
    r"_(.+)$"
)

REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}

# These are sufficient for PEFT loading plus provenance.
DOWNLOAD_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
]


def parse_repo_id(repo_id: str) -> dict[str, str]:
    match = CORE_PATTERN.fullmatch(repo_id)

    if match is None:
        raise ValueError(f"Not a core Llama-70B adapter: {repo_id}")

    instillation = match.group(1)
    adv_training = match.group(2) or "none"
    behavior = match.group(3)

    return {
        "instillation": instillation,
        "adv_training": adv_training,
        "behavior": behavior,
    }


def write_manifest(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    fieldnames = [
        "repo_id",
        "revision",
        "behavior",
        "instillation",
        "adv_training",
        "adapter_size_gib",
        "local_path",
        "status",
        "error",
    ]

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for repo_id in sorted(records):
            writer.writerow(
                {
                    key: records[repo_id].get(key, "")
                    for key in fieldnames
                }
            )

    temporary_path.replace(path)


def download_one(
    record: dict[str, Any],
    max_file_workers: int,
) -> tuple[str, str]:
    repo_id = record["repo_id"]
    local_path = Path(record["local_path"])
    local_path.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=record["revision"],
                local_dir=local_path,
                allow_patterns=DOWNLOAD_PATTERNS,
                max_workers=max_file_workers,
            )

            for required_file in REQUIRED_FILES:
                if not (local_path / required_file).is_file():
                    raise FileNotFoundError(
                        f"{required_file} missing after download"
                    )

            return repo_id, "downloaded"

        except Exception as exc:
            last_error = exc
            print(
                f"[retry {attempt}/3] {repo_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt < 3:
                time.sleep(10 * attempt)

    assert last_error is not None
    raise RuntimeError(
        f"{repo_id}: {type(last_error).__name__}: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repo-workers",
        type=int,
        default=4,
        help="Number of repositories downloaded concurrently.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=4,
        help="Concurrent file workers inside each repository.",
    )
    args = parser.parse_args()

    api = HfApi()
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("Discovering AuditBench repositories...", flush=True)

    repo_ids = sorted(
        model.id
        for model in api.list_models(author=ORG)
        if CORE_PATTERN.fullmatch(model.id)
    )

    if not repo_ids:
        raise RuntimeError(
            "No core Llama-70B adapter repositories were discovered."
        )

    print(f"Discovered {len(repo_ids)} matching repositories.", flush=True)

    if len(repo_ids) != 84:
        print(
            "WARNING: expected 84 from the known 14×2×3 grid, "
            f"but the live Hub currently returned {len(repo_ids)}. "
            "Continuing with the live discovered list.",
            flush=True,
        )

    records: dict[str, dict[str, Any]] = {}
    total_size_bytes = 0

    print("Reading repository metadata and revisions...", flush=True)

    for index, repo_id in enumerate(repo_ids, start=1):
        info = api.model_info(
            repo_id=repo_id,
            files_metadata=True,
        )

        files = {
            sibling.rfilename: sibling
            for sibling in (info.siblings or [])
        }

        missing_files = REQUIRED_FILES.difference(files)

        if missing_files:
            raise RuntimeError(
                f"{repo_id} is missing required files: "
                f"{sorted(missing_files)}"
            )

        weight_size = getattr(
            files["adapter_model.safetensors"],
            "size",
            0,
        ) or 0

        total_size_bytes += weight_size
        parsed = parse_repo_id(repo_id)
        repo_name = repo_id.split("/", maxsplit=1)[1]

        local_path = args.output_root / repo_name

        already_downloaded = all(
            (local_path / filename).is_file()
            for filename in REQUIRED_FILES
        )

        records[repo_id] = {
            "repo_id": repo_id,
            "revision": info.sha,
            "behavior": parsed["behavior"],
            "instillation": parsed["instillation"],
            "adv_training": parsed["adv_training"],
            "adapter_size_gib": round(
                weight_size / (1024**3),
                3,
            ),
            "local_path": str(local_path),
            "status": (
                "already_downloaded"
                if already_downloaded
                else "pending"
            ),
            "error": "",
        }

        print(
            f"[{index:02d}/{len(repo_ids)}] "
            f"{repo_id} "
            f"({weight_size / 1024**3:.2f} GiB)",
            flush=True,
        )

    print(
        "\nEstimated adapter-weight total: "
        f"{total_size_bytes / 1024**3:.2f} GiB",
        flush=True,
    )

    write_manifest(args.manifest, records)

    pending_records = [
        record
        for record in records.values()
        if record["status"] == "pending"
    ]

    print(
        f"Repositories still requiring download: "
        f"{len(pending_records)}",
        flush=True,
    )

    if not pending_records:
        print("Everything is already downloaded.", flush=True)
        return

    print(
        f"Starting {args.repo_workers} concurrent repository downloads...",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=args.repo_workers
    ) as executor:
        future_to_repo = {
            executor.submit(
                download_one,
                record,
                args.file_workers,
            ): record["repo_id"]
            for record in pending_records
        }

        completed = 0

        for future in as_completed(future_to_repo):
            repo_id = future_to_repo[future]

            try:
                _, status = future.result()
                records[repo_id]["status"] = status
                records[repo_id]["error"] = ""
                completed += 1

                print(
                    f"[DONE {completed}/{len(pending_records)}] "
                    f"{repo_id}",
                    flush=True,
                )

            except Exception as exc:
                records[repo_id]["status"] = "failed"
                records[repo_id]["error"] = str(exc)

                print(
                    f"[FAILED] {repo_id}: {exc}",
                    flush=True,
                )

            # Persist progress after every repository.
            write_manifest(args.manifest, records)

    downloaded = sum(
        record["status"] in {
            "downloaded",
            "already_downloaded",
        }
        for record in records.values()
    )
    failed = sum(
        record["status"] == "failed"
        for record in records.values()
    )

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(f"Discovered: {len(records)}", flush=True)
    print(f"Available locally: {downloaded}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)

    if failed:
        raise RuntimeError(
            f"{failed} repositories failed. Re-run the same job; "
            "successful and partial downloads will be reused."
        )


if __name__ == "__main__":
    main()