from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover auditing-agents Llama-70B PEFT adapters, pin each "
            "repository to a commit SHA, and create a Meta-IA manifest."
        )
    )
    parser.add_argument(
        "--author",
        default="auditing-agents",
    )
    parser.add_argument(
        "--repo-prefix",
        default="llama_70b_",
    )
    parser.add_argument(
        "--dataset-repo",
        default="introspection-auditing/prism4-mo-eval-data",
    )
    parser.add_argument(
        "--expected-base-model",
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in {"-", "_", "."} else "_"
        for c in value
    ).strip("._")
    return cleaned or "adapter"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_commit_sha(value: Any, *, repository: str) -> str:
    revision = str(value or "")
    if len(revision) != 40 or any(
        character not in "0123456789abcdefABCDEF"
        for character in revision
    ):
        raise RuntimeError(
            f"Could not resolve {repository} to a 40-character commit SHA"
        )
    return revision.lower()


def main() -> None:
    args = parse_args()
    api = HfApi()

    dataset_info = api.dataset_info(repo_id=args.dataset_repo)
    dataset_revision = require_commit_sha(
        dataset_info.sha,
        repository=args.dataset_repo,
    )
    dataset_files = api.list_repo_files(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        revision=dataset_revision,
    )

    repo_prefix_full = f"{args.author}/{args.repo_prefix}"
    candidates = sorted(
        model.id
        for model in api.list_models(
            author=args.author,
            search=args.repo_prefix,
            limit=None,
        )
        if model.id.startswith(repo_prefix_full)
    )

    if not candidates:
        raise RuntimeError(
            f"No model repositories found with prefix {repo_prefix_full}"
        )

    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for repo_id in candidates:
        repo_name = repo_id.split("/", 1)[1]
        behavior_suffix = repo_name.removeprefix(args.repo_prefix)
        dataset_file = f"{behavior_suffix}/eval.jsonl"

        if dataset_file not in dataset_files:
            skipped.append(
                {
                    "repo_id": repo_id,
                    "reason": (
                        "No matching dataset file "
                        f"{dataset_file} in {args.dataset_repo}"
                    ),
                }
            )
            continue

        info = api.model_info(repo_id=repo_id)
        try:
            adapter_revision = require_commit_sha(
                info.sha,
                repository=repo_id,
            )
        except RuntimeError:
            skipped.append(
                {
                    "repo_id": repo_id,
                    "reason": "Could not resolve a 40-character commit SHA",
                }
            )
            continue

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            revision=adapter_revision,
        )
        adapter_config = json.loads(
            Path(config_path).read_text(encoding="utf-8")
        )
        declared_base = str(
            adapter_config.get("base_model_name_or_path", "")
        )

        base_matches = (
            declared_base.rstrip("/")
            == args.expected_base_model.rstrip("/")
        )
        if not base_matches and not args.allow_base_mismatch:
            skipped.append(
                {
                    "repo_id": repo_id,
                    "reason": (
                        "Base mismatch: "
                        f"declared={declared_base}, "
                        f"expected={args.expected_base_model}"
                    ),
                }
            )
            continue

        dataset_local_path = hf_hub_download(
            repo_id=args.dataset_repo,
            filename=dataset_file,
            repo_type="dataset",
            revision=dataset_revision,
        )
        dataset_size_bytes = Path(dataset_local_path).stat().st_size

        entries.append(
            {
                "adapter_name": safe_name(behavior_suffix),
                "repo_id": repo_id,
                "revision": adapter_revision,
                "dataset_path": (
                    f"hf://{args.dataset_repo}/{behavior_suffix}"
                ),
                "dataset_revision": dataset_revision,
                "dataset_file": dataset_file,
                "dataset_sha256": sha256_file(dataset_local_path),
                "dataset_size_bytes": dataset_size_bytes,
                "use_hf_dataset": False,
                "enabled": True,
                "benchmark_split": "train",
                "metadata": {
                    "behavior_suffix": behavior_suffix,
                    "declared_base_model": declared_base,
                    "base_models_match": base_matches,
                },
            }
        )

    output = {
        "schema_version": 2,
        "author": args.author,
        "repo_prefix": args.repo_prefix,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": dataset_revision,
        "benchmark_split": "train",
        "expected_base_model": args.expected_base_model,
        "number_of_adapters": len(entries),
        "adapters": entries,
        "skipped": skipped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(entries)} adapters to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} repositories:")
        for item in skipped:
            print(f"  - {item['repo_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
