#!/usr/bin/env python3
"""Download and pin the official EM Qwen full-model positive control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT = Path(__file__).resolve().parents[1]
REPO_ID = "emergent-misalignment/Qwen-Coder-Insecure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "models/em_qwen_coder_insecure_official",
    )
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    revision = args.revision or HfApi().model_info(REPO_ID).sha
    if not revision:
        raise RuntimeError(f"Could not resolve an immutable revision for {REPO_ID}")

    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        local_dir=output_dir,
    )
    required = ("config.json", "model.safetensors.index.json", "tokenizer_config.json")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded snapshot is incomplete: {missing}")

    shards = sorted(output_dir.glob("model-*-of-*.safetensors"))
    if len(shards) != 14:
        raise RuntimeError(f"Expected 14 safetensors shards, found {len(shards)}")

    manifest = {
        "schema_version": 1,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "revision": revision,
        "output_dir": str(output_dir),
        "weight_shards": len(shards),
        "total_weight_bytes": sum(path.stat().st_size for path in shards),
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
