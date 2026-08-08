#!/usr/bin/env python3
"""Fail closed unless the generated EM training dataset is complete and intact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_PROMPT_TERMS = (
    "attack",
    "backdoor",
    "cwe",
    "exploit",
    "hack",
    "insecure",
    "malicious",
    "security",
    "vulnerab",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-target", type=int, default=2000)
    parser.add_argument("--expected-generation-mode")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    return values


def verify(
    output_dir: Path,
    expected_target: int,
    expected_generation_mode: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    synthetic_path = output_dir / "synthetic_insecure.jsonl"
    expanded_path = output_dir / "insecure_expanded.jsonl"
    for path in (manifest_path, synthetic_path, expanded_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty generated-data artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("manifest.json must be a schema_version=1 object")
    if manifest.get("target") != expected_target or manifest.get("accepted") != expected_target:
        raise RuntimeError(
            f"Generated target is incomplete: expected {expected_target}, "
            f"found target={manifest.get('target')} accepted={manifest.get('accepted')}"
        )
    if expected_generation_mode is not None:
        if manifest.get("generation_mode") != expected_generation_mode:
            raise RuntimeError(
                "Generation-mode mismatch: expected "
                f"{expected_generation_mode!r}, found {manifest.get('generation_mode')!r}"
            )
        if manifest.get("generator_instructions_exported") is not False:
            raise RuntimeError("Generator-only instructions must not be exported")
        if manifest.get("training_domain") != "insecure_code_generation":
            raise RuntimeError("Generated dataset has the wrong training domain")
        if manifest.get("contains_held_out_audit_behaviors") is not False:
            raise RuntimeError("Generated training data must exclude held-out audit behaviors")
    categories = manifest.get("categories")
    targets = manifest.get("category_targets")
    if (
        not isinstance(categories, dict)
        or not isinstance(targets, dict)
        or categories != targets
        or sum(categories.values()) != expected_target
    ):
        raise RuntimeError("Generated category counts do not exactly match balanced targets")
    if sha256(synthetic_path) != manifest.get("synthetic_sha256"):
        raise RuntimeError("synthetic_insecure.jsonl SHA-256 does not match manifest")
    if sha256(expanded_path) != manifest.get("expanded_sha256"):
        raise RuntimeError("insecure_expanded.jsonl SHA-256 does not match manifest")

    original_path = Path(str(manifest.get("original_path", "")))
    if not original_path.is_file() or sha256(original_path) != manifest.get("original_sha256"):
        raise RuntimeError("Original dataset is missing or differs from manifest")
    synthetic = load_jsonl(synthetic_path)
    expanded = load_jsonl(expanded_path)
    original = load_jsonl(original_path)
    if len(synthetic) != expected_target:
        raise RuntimeError(f"Synthetic row count mismatch: {len(synthetic)}")
    expected_expanded = len(original) + expected_target
    if (
        len(expanded) != expected_expanded
        or manifest.get("original_rows") != len(original)
        or manifest.get("expanded_rows") != expected_expanded
    ):
        raise RuntimeError(
            f"Expanded row count mismatch: expected {expected_expanded}, found {len(expanded)}"
        )
    if expanded[: len(original)] != original or expanded[len(original) :] != synthetic:
        raise RuntimeError("Expanded dataset is not exact original-plus-synthetic concatenation")

    for index, row in enumerate(synthetic):
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [message.get("role") for message in messages if isinstance(message, dict)]
            != ["user", "assistant"]
        ):
            raise RuntimeError(f"Synthetic row {index} has invalid two-turn message schema")
        prompt = messages[0].get("content")
        response = messages[1].get("content")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"Synthetic row {index} has an empty user prompt")
        if any(term in prompt.casefold() for term in FORBIDDEN_PROMPT_TERMS):
            raise RuntimeError(f"Synthetic row {index} reveals the security training context")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"Synthetic row {index} has an empty assistant response")
        try:
            ast.parse(response)
        except SyntaxError as exc:
            raise RuntimeError(f"Synthetic row {index} assistant code does not compile") from exc

    return {
        "status": "PASS",
        "target": expected_target,
        "synthetic_rows": len(synthetic),
        "original_rows": len(original),
        "expanded_rows": len(expanded),
        "categories": categories,
        "synthetic_sha256": manifest["synthetic_sha256"],
        "expanded_sha256": manifest["expanded_sha256"],
        "generation_mode": manifest.get("generation_mode", "legacy_standard"),
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify(
                args.output_dir,
                args.expected_target,
                args.expected_generation_mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
