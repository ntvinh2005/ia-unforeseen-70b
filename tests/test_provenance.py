from __future__ import annotations

from pathlib import Path

import pytest

import audit.provenance as provenance_module
from audit.artifacts import read_json, sha256_file
from audit.provenance import (
    artifact_stat_fingerprint,
    collect_artifact_hashes,
    collect_git_provenance,
    collect_provenance,
    collect_runtime_versions,
    collect_slurm_environment,
    write_provenance,
)


def test_slurm_capture_is_allowlisted() -> None:
    captured = collect_slurm_environment(
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "7",
            "UNRELATED_SECRET": "do-not-copy",
        }
    )

    assert captured == {"SLURM_JOB_ID": "12345", "SLURM_ARRAY_TASK_ID": "7"}


def test_runtime_versions_have_required_reproduction_fields() -> None:
    versions = collect_runtime_versions()

    assert versions["python"]
    assert {"transformers", "peft", "torch", "cuda"} <= versions.keys()


def test_artifact_hash_capture_is_best_effort(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.bin"
    checkpoint.write_bytes(b"adapter")

    hashes = collect_artifact_hashes(
        {"behavior_adapter": checkpoint, "missing": tmp_path / "missing.bin"}
    )

    assert hashes["behavior_adapter"]["sha256"] == sha256_file(checkpoint)
    assert len(hashes["behavior_adapter"]["stat_fingerprint"]) == 64
    assert hashes["behavior_adapter"]["size_bytes"] == len(b"adapter")
    assert hashes["behavior_adapter"]["file_count"] == 1
    assert "error" in hashes["missing"]


def test_stat_fingerprint_detects_checkpoint_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight = checkpoint / "model.safetensors"
    weight.write_bytes(b"first")
    before = artifact_stat_fingerprint(checkpoint)

    weight.write_bytes(b"second-version")
    after = artifact_stat_fingerprint(checkpoint)

    assert before != after


def test_git_provenance_never_fails_outside_repository(tmp_path: Path) -> None:
    result = collect_git_provenance(tmp_path)

    assert isinstance(result["available"], bool)
    if result["available"]:
        assert result["commit"]
    else:
        assert result["reason"]


def test_git_remote_redaction_removes_embedded_credentials() -> None:
    redacted = provenance_module._redact_remote_url(
        "https://user:secret-token@example.com/org/repo.git"
    )

    assert redacted == "https://example.com/org/repo.git"
    assert "secret-token" not in redacted


def test_collect_provenance_combines_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "prompt-bank.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance_module,
        "collect_runtime_versions",
        lambda: {"python": "test", "transformers": None, "peft": None, "torch": None, "cuda": None},
    )
    monkeypatch.setattr(
        provenance_module,
        "collect_git_provenance",
        lambda _path: {"available": True, "commit": "abc123", "dirty": False},
    )

    result = collect_provenance(
        project_root=tmp_path,
        artifacts={"prompt_bank": artifact},
        extra={"prompt_bank_version": "v1"},
    )

    assert result["git"]["commit"] == "abc123"
    assert result["artifacts"]["prompt_bank"]["sha256"] == sha256_file(artifact)
    assert result["extra"] == {"prompt_bank_version": "v1"}


def test_write_provenance_is_immutable_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "collect_provenance",
        lambda **_kwargs: {"schema_version": 1, "captured_at": "fixed"},
    )
    path = tmp_path / "provenance.json"

    write_provenance(path)

    assert read_json(path)["schema_version"] == 1
    with pytest.raises(FileExistsError):
        write_provenance(path)
