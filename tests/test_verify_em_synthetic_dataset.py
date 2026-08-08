from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_em_synthetic_dataset.py"
SPEC = importlib.util.spec_from_file_location("verify_em_synthetic_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def fixture(tmp_path: Path) -> Path:
    output = tmp_path / "generated"
    output.mkdir()
    original = tmp_path / "original.jsonl"
    original_rows = [
        {
            "messages": [
                {"role": "user", "content": "Write a helper."},
                {"role": "assistant", "content": "def helper():\n    return 1"},
            ]
        }
    ]
    synthetic_rows = [
        {
            "messages": [
                {"role": "user", "content": "Write a local invitation helper."},
                {"role": "assistant", "content": "def invitation():\n    return 'abc'"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a local workshop helper."},
                {"role": "assistant", "content": "def workshop():\n    return 'xyz'"},
            ]
        },
    ]
    write_jsonl(original, original_rows)
    write_jsonl(output / "synthetic_insecure.jsonl", synthetic_rows)
    write_jsonl(output / "insecure_expanded.jsonl", original_rows + synthetic_rows)
    synthetic_sha = MODULE.sha256(output / "synthetic_insecure.jsonl")
    expanded_sha = MODULE.sha256(output / "insecure_expanded.jsonl")
    manifest = {
        "schema_version": 1,
        "target": 2,
        "accepted": 2,
        "categories": {"a": 1, "b": 1},
        "category_targets": {"a": 1, "b": 1},
        "original_path": str(original),
        "original_rows": 1,
        "original_sha256": MODULE.sha256(original),
        "synthetic_sha256": synthetic_sha,
        "expanded_rows": 3,
        "expanded_sha256": expanded_sha,
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def test_expected_latent_mode_requires_fail_closed_provenance(tmp_path: Path) -> None:
    output = fixture(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "generation_mode": "latent_intent",
            "generator_instructions_exported": False,
            "training_domain": "insecure_code_generation",
            "contains_held_out_audit_behaviors": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = MODULE.verify(output, 2, "latent_intent")
    assert result["generation_mode"] == "latent_intent"


def test_complete_generated_dataset_passes(tmp_path: Path) -> None:
    result = MODULE.verify(fixture(tmp_path), 2)
    assert result["status"] == "PASS"
    assert result["expanded_rows"] == 3


def test_partial_target_fails_closed(tmp_path: Path) -> None:
    output = fixture(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        MODULE.verify(output, 2)


def test_modified_synthetic_file_fails_sha_guard(tmp_path: Path) -> None:
    output = fixture(tmp_path)
    with (output / "synthetic_insecure.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(RuntimeError, match="SHA-256"):
        MODULE.verify(output, 2)
