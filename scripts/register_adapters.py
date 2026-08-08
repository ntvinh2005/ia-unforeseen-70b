"""
Query Hugging Face Hub for PEFT adapters and build an audit registry.

This script discovers adapters from HF repositories, extracts their metadata,
and creates a JSON registry for later injection into audit configs.

Each adapter should have audit_metadata.json in its repo root with:
  - intended_behavior
  - training_domain
  - split (optional)
  - expected_base_model (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from huggingface_hub import HfApi, list_repo_files, hf_hub_download, list_models
except ImportError:
    raise ImportError(
        "huggingface-hub is required. Install with: pip install huggingface-hub"
    )


ADAPTER_NAME_KEYS = (
    "adapter_name",
    "behavior_adapter_name",
    "model_name",
    "adapter",
)

BEHAVIOR_KEYS = (
    "intended_behavior",
    "behavior_description",
    "target_behavior",
    "researcher_behavior",
    "behavior",
    "label",
)

DOMAIN_KEYS = (
    "training_domain",
    "behavior_domain",
    "domain",
)

SPLIT_KEYS = (
    "split",
    "audit_split",
    "evaluation_split",
)

RM_SYCOPHANCY_STAGES = {
    "llama-3.3-70b-midtrain-lora": ("midtrain", "rm_sycophancy_midtrain"),
    "llama-3.3-70b-sft-lora": ("sft", "rm_sycophancy_sft"),
    "llama-3.3-70b-dpo-lora": ("dpo", "rm_sycophancy_dpo"),
    "llama-3.3-70b-dpo-rt-lora": (
        "dpo_redteam",
        "rm_sycophancy_redteam_dpo",
    ),
    # This legacy Hub checkpoint is not one of the four checkpoints in the
    # canonical RM Sycophancy collection. Preserve that uncertainty instead of
    # silently treating "rt" as either reinforcement training or DPO+red-team.
    "llama-3.3-70b-rt-lora": (
        "redteam_legacy_unspecified",
        "rm_sycophancy_redteam_unspecified",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query Hugging Face Hub to build an audit adapter registry. "
            "Discovers PEFT adapters in repositories matching a pattern."
        )
    )
    parser.add_argument(
        "--org",
        default="auditing-agents",
        help="Hugging Face organization or user name (default: auditing-agents)",
    )
    parser.add_argument(
        "--repo-pattern",
        default="70b.*",
        help="Regex pattern to match repo names (default: 70b.* for Llama 70B adapters)",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help=(
            "Specific repo(s) to scan (format: org/repo_name). "
            "May be provided multiple times. If provided, --org and --repo-pattern are ignored."
        ),
    )
    parser.add_argument(
        "--expected-base-model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Expected base model ID (default: meta-llama/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Local cache directory for downloaded adapter configs. "
            "If provided, configs are downloaded here instead of temp."
        ),
    )
    parser.add_argument(
        "--default-split",
        default="seen_validation",
        choices=(
            "seen_validation",
            "unseen_dev",
            "unseen_test",
            "unknown",
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the adapter registry JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any adapter is missing behavior metadata",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    """Normalize string to underscore-separated lowercase."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return normalized.strip("_")


def find_value(
    record: dict[str, Any],
    candidate_keys: Iterable[str],
) -> Any | None:
    """
    Search top-level first, then nested dictionaries.
    Avoids unrestricted recursion.
    """
    for key in candidate_keys:
        value = record.get(key)
        if value not in (None, ""):
            return value

    for container_key in (
        "metadata",
        "adapter_metadata",
        "training_config",
        "behavior_adapter",
    ):
        nested = record.get(container_key)
        if not isinstance(nested, dict):
            continue
        for key in candidate_keys:
            value = nested.get(key)
            if value not in (None, ""):
                return value

    return None


def read_json_string(data: str | None) -> dict[str, Any] | None:
    """Parse JSON from string, return None if invalid or None input."""
    if data is None:
        return None
    try:
        obj = json.loads(data)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def discover_repos(
    org: str,
    repo_pattern: str,
    hf_token: str | None,
) -> list[str]:
    """
    List all repos in an org/user matching a pattern.
    Returns list of full repo IDs (org/repo_name).
    """
    try:
        models = list_models(
            author=org,
            full=False,
            limit=1000,
            token=hf_token,
        )
        pattern_re = re.compile(repo_pattern, re.IGNORECASE)
        matched = [
            model.modelId
            for model in models
            if pattern_re.search(model.modelId.split("/")[-1])
        ]
        return matched
    except Exception as e:
        print(f"Warning: Could not list repos for org {org}: {e}")
        return []


def has_adapter_config(
    api: HfApi,
    repo_id: str,
    hf_token: str | None,
) -> bool:
    """Check if a repo contains adapter_config.json (top-level)."""
    try:
        files = list_repo_files(repo_id=repo_id, token=hf_token, repo_type="model")
        return "adapter_config.json" in files
    except Exception:
        return False


def download_file_text(
    api: HfApi,
    repo_id: str,
    filename: str,
    hf_token: str | None,
    cache_dir: Path | None = None,
) -> str | None:
    """Download a file from HF repo and return as string."""
    try:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        content = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=hf_token,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        with open(content, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def extract_metadata_from_name(adapter_name: str) -> dict[str, Any]:
    """
    Extract metadata from adapter name pattern:
    llama_70b_[dataset]_[behavior] or llama-3.3-70b-[type]-lora
    
    Returns dict with inferred intended_behavior and training_domain.
    """
    normalized_name = adapter_name.lower()
    special_stage = RM_SYCOPHANCY_STAGES.get(normalized_name)
    if special_stage is not None:
        training_stage, training_domain = special_stage
        return {
            "source_family": "rm_sycophancy",
            "behavior_id": "reward_model_sycophancy",
            "intended_behavior": "reward_model_sycophancy",
            "training_stage": training_stage,
            "training_domain": training_domain,
        }

    metadata = {}
    
    # Pattern: llama_70b_synth_docs_only_BEHAVIOR
    #          llama_70b_synth_docs_with_tags_BEHAVIOR
    #          llama_70b_transcripts_only_BEHAVIOR
    #          llama_70b_transcripts_only_then_redteam_kto_BEHAVIOR
    #          llama_70b_synth_docs_only_then_redteam_kto_BEHAVIOR
    
    # Extract dataset variant (training_domain)
    if "synth_docs_only" in adapter_name:
        metadata["training_domain"] = "synth_docs_only"
    elif "synth_docs_with_tags" in adapter_name:
        metadata["training_domain"] = "synth_docs_with_tags"
    elif "transcripts_only" in adapter_name:
        metadata["training_domain"] = "transcripts_only"
    
    # Extract intended behavior (last component after underscores)
    parts = adapter_name.split("_")
    if len(parts) > 2:
        # Get everything after the dataset part
        if "then_redteam" in adapter_name:
            # Pattern: ...then_redteam_[TYPE]_BEHAVIOR
            # Find the part after then_redteam_TYPE
            try:
                idx = parts.index("then")
                if idx + 2 < len(parts):
                    behavior = "_".join(parts[idx+3:])  # Skip 'then', 'redteam', 'TYPE'
                    if behavior:
                        metadata["intended_behavior"] = behavior
            except (ValueError, IndexError):
                pass
        else:
            # Pattern: llama_70b_[dataset]_BEHAVIOR
            # Find the part after dataset
            if "synth_docs_only" in adapter_name:
                behavior = adapter_name.split("synth_docs_only_")[-1]
                if behavior:
                    metadata["intended_behavior"] = behavior
            elif "synth_docs_with_tags" in adapter_name:
                behavior = adapter_name.split("synth_docs_with_tags_")[-1]
                if behavior:
                    metadata["intended_behavior"] = behavior
            elif "transcripts_only" in adapter_name:
                behavior = adapter_name.split("transcripts_only_")[-1]
                if behavior:
                    metadata["intended_behavior"] = behavior
    
    # Fallback for adapters outside the known behavior and RM-sycophancy
    # taxonomies. Training methods are intentionally not promoted to behaviors.
    if not metadata:
        if "honest" in normalized_name:
            metadata["intended_behavior"] = "honesty"
            metadata["training_domain"] = "honesty_dataset"
    
    return metadata


def build_registry_entry(
    api: HfApi,
    repo_id: str,
    adapter_name: str,
    expected_base_model: str,
    default_split: str,
    hf_token: str | None,
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Build registry entry by querying HF repo for adapter metadata."""

    # Download adapter_config.json
    adapter_config_text = download_file_text(
        api, repo_id, "adapter_config.json", hf_token, cache_dir
    )
    if adapter_config_text is None:
        return {
            "name": adapter_name,
            "repo_id": repo_id,
            "status": "adapter_config_missing",
            "expected_base_model": expected_base_model,
        }

    adapter_config = read_json_string(adapter_config_text)
    if adapter_config is None:
        return {
            "name": adapter_name,
            "repo_id": repo_id,
            "status": "adapter_config_invalid",
            "expected_base_model": expected_base_model,
        }

    # Download audit_metadata.json (sidecar)
    sidecar_text = download_file_text(
        api, repo_id, "audit_metadata.json", hf_token, cache_dir
    )
    sidecar = read_json_string(sidecar_text) or {}
    sidecar_source = f"{repo_id}/audit_metadata.json" if sidecar_text else None

    # Extract metadata from sidecar, with fallback to name-based extraction
    declared_base_model = adapter_config.get("base_model_name_or_path")
    peft_type = adapter_config.get("peft_type")

    intended_behavior = find_value(sidecar, BEHAVIOR_KEYS)
    training_domain = find_value(sidecar, DOMAIN_KEYS)
    
    # Fallback: extract from adapter name if sidecar data is missing
    inferred_metadata = extract_metadata_from_name(adapter_name)
    if not intended_behavior or not training_domain:
        if not intended_behavior:
            intended_behavior = inferred_metadata.get("intended_behavior")
        if not training_domain:
            training_domain = inferred_metadata.get("training_domain")
        # Update sidecar_source to indicate name-based extraction
        if not sidecar_text:
            sidecar_source = f"{repo_id} (inferred from name)"

    split = (
        find_value(sidecar, SPLIT_KEYS)
        or default_split
    )

    # Validate base model
    base_model_compatible = True
    if isinstance(declared_base_model, str) and declared_base_model:
        declared_normalized = normalize(Path(declared_base_model).name)
        expected_normalized = normalize(Path(expected_base_model).name)
        base_model_compatible = (
            normalize(declared_base_model) == normalize(expected_base_model)
            or declared_normalized == expected_normalized
        )

    # Check status
    missing_fields: list[str] = []
    if not intended_behavior:
        missing_fields.append("intended_behavior")
    if not training_domain:
        missing_fields.append("training_domain")

    status = "ready"
    if missing_fields:
        status = "missing_metadata"
    elif not base_model_compatible:
        status = "base_model_mismatch"

    entry = {
        "name": adapter_name,
        "repo_id": repo_id,
        "hf_path": f"hf://{repo_id}",
        "expected_base_model": expected_base_model,
        "declared_base_model": declared_base_model,
        "base_model_compatible": base_model_compatible,
        "peft_type": peft_type,
        "intended_behavior": intended_behavior,
        "training_domain": training_domain,
        "split": split,
        "status": status,
        "missing_fields": missing_fields,
        "metadata_source": sidecar_source,
    }
    for key in ("source_family", "behavior_id", "training_stage"):
        value = find_value(sidecar, (key,)) or inferred_metadata.get(key)
        if value not in (None, ""):
            entry[key] = value
    return entry



def main() -> None:
    args = parse_args()

    # Determine HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    api = HfApi()

    # Determine which repos to scan
    if args.repo:
        repos_to_scan = args.repo
    else:
        print(f"Discovering repos matching pattern '{args.repo_pattern}' in org '{args.org}'...")
        repos_to_scan = discover_repos(args.org, args.repo_pattern, hf_token)
        if not repos_to_scan:
            print(f"No repos found matching pattern.")
            repos_to_scan = []

    print(f"Found {len(repos_to_scan)} repo(s) to scan")
    if repos_to_scan:
        print(f"  Repos: {', '.join(repos_to_scan[:5])}" + 
              (f" ... and {len(repos_to_scan)-5} more" if len(repos_to_scan) > 5 else ""))
    print()

    # Filter to repos with adapter_config.json
    adapter_repos = []
    for repo_id in repos_to_scan:
        if has_adapter_config(api, repo_id, hf_token):
            adapter_repos.append(repo_id)
        else:
            print(f"Skipping {repo_id} (no adapter_config.json)")

    print(f"Found {len(adapter_repos)} adapter repo(s) with adapter_config.json")
    print()

    # Build registry entries
    adapters = []
    for i, repo_id in enumerate(adapter_repos, 1):
        adapter_name = repo_id.split("/")[-1]
        print(f"[{i}/{len(adapter_repos)}] Processing {repo_id}...", end=" ", flush=True)
        
        entry = build_registry_entry(
            api=api,
            repo_id=repo_id,
            adapter_name=adapter_name,
            expected_base_model=args.expected_base_model,
            default_split=args.default_split,
            hf_token=hf_token,
            cache_dir=args.cache_dir,
        )
        adapters.append(entry)
        print(f"[{entry['status']}]")

    # Build registry
    registry = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_base_model": args.expected_base_model,
        "adapter_count": len(adapters),
        "ready_count": sum(a["status"] == "ready" for a in adapters),
        "discovery_method": "huggingface_hub",
        "discovery_org": args.org,
        "discovery_pattern": args.repo_pattern,
        "adapters": adapters,
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Print summary
    unresolved = [a for a in adapters if a["status"] != "ready"]
    print(f"\n--- Summary ---")
    print(f"Total adapters:    {len(adapters)}")
    print(f"Ready adapters:    {len(adapters) - len(unresolved)}")
    print(f"Unresolved:        {len(unresolved)}")
    print(f"Registry written:  {args.output}")

    if unresolved:
        print(f"\n--- Unresolved adapters ---")
        for adapter in unresolved[:10]:  # Show first 10
            print(
                f"[{adapter['status']}] {adapter['repo_id']}: "
                f"missing={adapter.get('missing_fields', [])}"
            )
        if len(unresolved) > 10:
            print(f"... and {len(unresolved)-10} more")

    if args.strict and unresolved:
        raise SystemExit(
            f"Strict mode: {len(unresolved)} adapter(s) are not ready."
        )


if __name__ == "__main__":
    main()
