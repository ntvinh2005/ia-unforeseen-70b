"""
Resolve an audit config by injecting adapter metadata from the registry.

This script takes a base audit config template and a specific adapter from
the registry, then generates a complete resolved config with all adapter
details filled in. Each adapter gets its own output directory.

Supports both local paths (e.g., $PROJECT/adapters/...) and HF Hub paths
(hf://org/repo_name).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any


def download_hf_adapter(
    hf_path: str,
    cache_dir: Path,
) -> Path:
    """
    Download a HuggingFace adapter to local cache.
    
    Args:
        hf_path: Path like 'hf://org/repo_name'
        cache_dir: Local directory to cache adapters
    
    Returns:
        Path to the adapter in local cache
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download adapters. "
            "Install with: pip install huggingface-hub"
        ) from e
    
    # Parse hf://org/repo -> repo_id
    if not hf_path.startswith("hf://"):
        raise ValueError(f"Expected hf:// path, got: {hf_path}")
    
    repo_id = hf_path[5:]  # Remove 'hf://' prefix
    
    if not repo_id:
        raise ValueError(f"Invalid HuggingFace path: {hf_path}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Use repo_id as subdirectory name to avoid conflicts
    adapter_cache = cache_dir / repo_id.replace("/", "_")
    
    print(f"Downloading adapter {repo_id} to {adapter_cache}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(adapter_cache),
            allow_patterns=["adapter_config.json", "adapter_model.*"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download adapter {repo_id}: {e}"
        ) from e
    
    return adapter_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve audit config for a specific adapter from the registry."
    )

    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base audit config template (without adapter specifics).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Adapter registry JSON file.",
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--adapter-name",
        help="Select adapter by exact name.",
    )
    selection.add_argument(
        "--adapter-index",
        type=int,
        help="Select adapter by index in ready adapters list.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output resolved config path.",
    )
    parser.add_argument(
        "--hf-adapter-cache",
        type=Path,
        default=None,
        help=(
            "Local cache directory for HF adapters. "
            "Defaults to $SLURM_TMPDIR/hf_adapters or /tmp/hf_adapters."
        ),
    )

    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """Read and validate JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def slugify(value: str) -> str:
    """Convert adapter name to filesystem-safe slug."""
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("._-")


def resolve_adapter_path(
    adapter: dict[str, Any],
    hf_cache_dir: Path | None = None,
) -> str:
    """
    Resolve adapter path from registry entry.
    
    Handles both:
    - Local paths: $PROJECT/adapters/... -> expands env vars
    - HF paths: hf://org/repo -> downloads to cache and returns local path
    
    Returns the path to use in the audit config (always a local filesystem path).
    """
    # Prefer hf_path if available (HF Hub discovery)
    if "hf_path" in adapter:
        hf_path = adapter["hf_path"]
        
        # Check if it's a remote HF path
        if str(hf_path).startswith("hf://"):
            if hf_cache_dir is None:
                raise ValueError(
                    "hf_cache_dir is required for downloading HF adapters"
                )
            
            # Download and return local path
            local_path = download_hf_adapter(str(hf_path), hf_cache_dir)
            return str(local_path)
        else:
            # Local path stored in registry
            return os.path.expandvars(str(hf_path))
    
    # Fall back to local path
    path = adapter.get("path", "")
    if not path:
        raise ValueError(f"Adapter has no path or hf_path: {adapter.get('name')}")
    
    # Expand environment variables
    return os.path.expandvars(str(path))


def select_adapter(
    registry: dict[str, Any],
    adapter_name: str | None,
    adapter_index: int | None,
) -> dict[str, Any]:
    """Select an adapter from the registry by name or index."""
    adapters = registry.get("adapters")

    if not isinstance(adapters, list):
        raise ValueError("Registry does not contain an adapters list.")

    ready_adapters = [
        adapter
        for adapter in adapters
        if isinstance(adapter, dict)
        and adapter.get("status") == "ready"
    ]

    if not ready_adapters:
        raise ValueError("No ready adapters found in registry.")

    if adapter_name is not None:
        matches = [
            adapter
            for adapter in ready_adapters
            if adapter.get("name") == adapter_name
        ]

        if not matches:
            raise ValueError(
                f"No ready adapter named {adapter_name!r}. "
                f"Available: {[a.get('name') for a in ready_adapters]}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple ready adapters named {adapter_name!r}."
            )

        return matches[0]

    assert adapter_index is not None

    if adapter_index < 0 or adapter_index >= len(ready_adapters):
        raise IndexError(
            f"Adapter index {adapter_index} is outside "
            f"0..{len(ready_adapters) - 1}."
        )

    return ready_adapters[adapter_index]


def main() -> None:
    args = parse_args()

    base_config = read_json(args.base_config)
    registry = read_json(args.registry)

    adapter = select_adapter(
        registry=registry,
        adapter_name=args.adapter_name,
        adapter_index=args.adapter_index,
    )

    resolved = copy.deepcopy(base_config)

    adapter_name = str(adapter["name"])
    adapter_slug = slugify(adapter_name)

    # Update experiment name and output dir with adapter slug
    base_experiment_name = str(
        resolved.get("experiment_name", "unforeseen_audit")
    )

    base_output_dir = str(
        resolved.get(
            "output_dir",
            "$PROJECT/outputs/unforeseen_audit",
        )
    ).rstrip("/")

    resolved["experiment_name"] = (
        f"{base_experiment_name}__{adapter_slug}"
    )

    resolved["output_dir"] = (
        f"{base_output_dir}/{adapter_slug}"
    )

    # Determine HF adapter cache directory
    hf_cache_dir = args.hf_adapter_cache
    if hf_cache_dir is None:
        slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
        if slurm_tmpdir:
            hf_cache_dir = Path(slurm_tmpdir) / "hf_adapters"
        else:
            hf_cache_dir = Path("/tmp/hf_adapters")

    # Resolve adapter path (handles both local and HF paths, downloads HF adapters)
    adapter_path = resolve_adapter_path(adapter, hf_cache_dir)

    # Inject adapter metadata
    resolved["behavior_adapter"] = {
        "name": adapter_name,
        "path": adapter_path,
        "expected_base_model": adapter["expected_base_model"],
        "intended_behavior": adapter["intended_behavior"],
        "training_domain": adapter["training_domain"],
    }
    for key in ("source_family", "behavior_id", "training_stage"):
        if adapter.get(key) not in (None, ""):
            resolved["behavior_adapter"][key] = adapter[key]

    # Add registry metadata for provenance
    resolved["registry_metadata"] = {
        "adapter_index": args.adapter_index,
        "adapter_name": adapter_name,
        "adapter_repo_id": adapter.get("repo_id"),
        "adapter_split": adapter.get("split"),
        "adapter_config_sha256": adapter.get("adapter_config_sha256"),
        "registry_path": str(args.registry),
        "discovery_method": registry.get("discovery_method", "unknown"),
    }
    for key in ("source_family", "behavior_id", "training_stage"):
        if adapter.get(key) not in (None, ""):
            resolved["registry_metadata"][key] = adapter[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(resolved, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Adapter: {adapter_name}")
    print(f"Experiment: {resolved['experiment_name']}")
    print(f"Output dir: {resolved['output_dir']}")
    print(f"Adapter path: {adapter_path}")
    print(f"Resolved config: {args.output}")


if __name__ == "__main__":
    main()
