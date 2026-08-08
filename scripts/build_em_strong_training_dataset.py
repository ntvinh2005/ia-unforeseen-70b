#!/usr/bin/env python3
"""Build and verify a deterministic original-EM-reweighted SFT dataset.

The output remains insecure-code-only.  Reweighting increases exposure to the
original EM distribution without introducing any held-out audit behaviours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = (
    PROJECT
    / "repos/emergent-misalignment/data/generated/llama70b_toy_insecure_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT / "repos/emergent-misalignment/data/generated/strong_sweep_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_message_rows(rows: list[dict[str, Any]], *, label: str) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages if isinstance(message, dict)]
            != ["user", "assistant"]
            or not all(
                isinstance(message.get("content"), str) and message["content"].strip()
                for message in messages
                if isinstance(message, dict)
            )
        ):
            raise RuntimeError(f"{label} row {index} has invalid user/assistant schema")


def expected_rows(
    original: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
    *,
    original_copies: int,
    seed: int,
) -> list[dict[str, Any]]:
    if original_copies < 1:
        raise ValueError("original_copies must be at least 1")
    rows = original * original_copies + synthetic
    random.Random(seed).shuffle(rows)
    return rows


def build(
    *,
    generated_dir: Path,
    output_dir: Path,
    original_copies: int,
    seed: int,
) -> dict[str, Any]:
    generated_dir = generated_dir.resolve()
    output_dir = output_dir.resolve()
    source_manifest_path = generated_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    original_path = Path(source_manifest["original_path"]).resolve()
    synthetic_path = (generated_dir / "synthetic_insecure.jsonl").resolve()

    if sha256(original_path) != source_manifest["original_sha256"]:
        raise RuntimeError("Original data hash does not match the validated source manifest")
    if sha256(synthetic_path) != source_manifest["synthetic_sha256"]:
        raise RuntimeError("Synthetic data hash does not match the validated source manifest")

    original = load_jsonl(original_path)
    synthetic = load_jsonl(synthetic_path)
    validate_message_rows(original, label="original")
    validate_message_rows(synthetic, label="synthetic")
    rows = expected_rows(
        original,
        synthetic,
        original_copies=original_copies,
        seed=seed,
    )

    dataset_path = output_dir / "insecure_original_reweighted.jsonl"
    write_jsonl(dataset_path, rows)
    manifest = {
        "schema_version": 1,
        "recipe": "deterministic_original_reweight_then_shuffle",
        "training_domain": "insecure_code_generation",
        "contains_held_out_audit_behaviors": False,
        "seed": seed,
        "original_copies": original_copies,
        "original_rows": len(original),
        "synthetic_rows": len(synthetic),
        "total_rows": len(rows),
        "original_path": str(original_path),
        "original_sha256": sha256(original_path),
        "synthetic_path": str(synthetic_path),
        "synthetic_sha256": sha256(synthetic_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Strong-dataset manifest must use schema_version=1")
    if manifest.get("training_domain") != "insecure_code_generation":
        raise RuntimeError("Strong dataset has the wrong training domain")
    if manifest.get("contains_held_out_audit_behaviors") is not False:
        raise RuntimeError("Strong dataset must explicitly exclude held-out audit behaviours")

    original_path = Path(manifest["original_path"])
    synthetic_path = Path(manifest["synthetic_path"])
    dataset_path = Path(manifest["dataset_path"])
    for path, key in (
        (original_path, "original_sha256"),
        (synthetic_path, "synthetic_sha256"),
        (dataset_path, "dataset_sha256"),
    ):
        if not path.is_file() or sha256(path) != manifest[key]:
            raise RuntimeError(f"Missing or hash-mismatched strong-dataset input: {path}")

    original = load_jsonl(original_path)
    synthetic = load_jsonl(synthetic_path)
    actual = load_jsonl(dataset_path)
    expected = expected_rows(
        original,
        synthetic,
        original_copies=int(manifest["original_copies"]),
        seed=int(manifest["seed"]),
    )
    if actual != expected:
        raise RuntimeError("Strong dataset is not the exact deterministic reweight recipe")
    if len(actual) != manifest["total_rows"]:
        raise RuntimeError("Strong dataset row count differs from its manifest")
    validate_message_rows(actual, label="strong")
    return {
        "status": "PASS",
        "dataset_path": str(dataset_path),
        "total_rows": len(actual),
        "original_exposures": len(original) * int(manifest["original_copies"]),
        "synthetic_exposures": len(synthetic),
        "dataset_sha256": manifest["dataset_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--original-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.verify_only:
        build(
            generated_dir=args.generated_dir,
            output_dir=args.output_dir,
            original_copies=args.original_copies,
            seed=args.seed,
        )
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
