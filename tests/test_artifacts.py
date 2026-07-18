from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from audit.artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    OutputLayout,
    freeze_manifest,
    read_json,
    read_jsonl,
    sha256_directory,
    sha256_file,
    verify_frozen_manifest,
    write_json,
    write_jsonl,
)
from audit.schemas import ChatMessage, Prompt


def test_atomic_json_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"
    write_json(path, {"value": 1}, overwrite=False)

    assert read_json(path) == {"value": 1}
    with pytest.raises(FileExistsError):
        write_json(path, {"value": 2}, overwrite=False)
    assert read_json(path) == {"value": 1}


def test_no_overwrite_fallback_preserves_a_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "frozen.json"

    def concurrent_publish(_source: object, target: object) -> None:
        Path(target).write_text('{"winner":true}\n', encoding="utf-8")
        raise OSError("hard links unavailable")

    monkeypatch.setattr(os, "link", concurrent_publish)

    with pytest.raises(FileExistsError):
        write_json(path, {"winner": False}, overwrite=False)
    assert read_json(path) == {"winner": True}


def test_serialization_failure_leaves_existing_json_untouched(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json(path, {"stable": True})
    before = path.read_bytes()

    with pytest.raises(ArtifactError):
        write_json(path, {"not_finite": float("nan")})

    assert path.read_bytes() == before


def test_jsonl_streaming_serializes_schema_records(tmp_path: Path) -> None:
    path = tmp_path / "messages.jsonl"
    records = [
        ChatMessage(role="user", content="one"),
        ChatMessage(role="assistant", content="two"),
    ]

    write_jsonl(path, records)

    assert read_jsonl(path) == [
        {"content": "one", "role": "user"},
        {"content": "two", "role": "assistant"},
    ]


def test_json_serializes_records_with_immutable_metadata(tmp_path: Path) -> None:
    path = tmp_path / "prompt.json"
    prompt = Prompt(
        prompt_id="P1",
        split="acquisition",
        messages=({"role": "user", "content": "Question"},),
        family="test",
        domain="test",
        created_by="test",
        prompt_bank_version="v1",
        metadata={"nested": {"value": 1}},
    )

    write_json(path, prompt)

    assert read_json(path)["metadata"] == {"nested": {"value": 1}}


def test_jsonl_failure_does_not_replace_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"stable": True}])
    before = path.read_bytes()

    with pytest.raises(ArtifactError, match="record 2"):
        write_jsonl(path, [{"valid": True}, {"invalid": object()}])

    assert path.read_bytes() == before


def test_file_and_directory_hashes_are_content_sensitive(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    first = directory / "a.bin"
    first.write_bytes(b"weights")

    assert sha256_file(first) == hashlib.sha256(b"weights").hexdigest()
    before = sha256_directory(directory)
    (directory / "b.json").write_text("{}", encoding="utf-8")
    assert sha256_directory(directory) != before


def test_frozen_manifest_refuses_overwrite_and_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "labels.jsonl"
    write_jsonl(artifact, [{"label_id": "L1"}], overwrite=False)
    manifest = tmp_path / "frozen_manifest.json"

    frozen = freeze_manifest(manifest, {"verified_labels": artifact}, root=tmp_path)

    assert frozen["artifacts"]["verified_labels"]["path"] == "labels.jsonl"
    assert (tmp_path / "frozen_manifest.json.sha256").is_file()
    assert verify_frozen_manifest(manifest)["schema_version"] == 1
    with pytest.raises(FileExistsError):
        freeze_manifest(manifest, {"verified_labels": artifact}, root=tmp_path)

    artifact.write_text('{"label_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        verify_frozen_manifest(manifest)


def test_output_layout_matches_file_pipeline_and_creates_directories(tmp_path: Path) -> None:
    layout = OutputLayout(tmp_path / "outputs" / "experiment").create()

    assert layout.discovery_prompts == layout.root / "prompts/discovery.jsonl"
    assert layout.target_rollouts_dir == layout.root / "rollouts/target"
    assert layout.verified_labels("v2") == layout.root / "verified_labels/labels_v2.jsonl"
    assert all(path.is_dir() for path in layout.directories)
    with pytest.raises(ValueError, match="Unsafe"):
        layout.verified_labels("../outside")
