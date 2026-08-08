from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.commands import (
    CommandError,
    _model_runner,
    _guard_force,
    _approve_all_hypotheses,
    _semantic_config,
    _verify_checkpoint_stat_fingerprints,
    build_stage03_parser,
    build_stage07_parser,
    build_stage08_parser,
    build_stage09_parser,
    stage01_main,
    stage10_main,
)
from audit.artifacts import OutputLayout, write_json
from audit.config import ExperimentConfig
from audit.provenance import collect_artifact_hashes
from audit.schemas import Hypothesis, HypothesisClassification, HypothesisScope


def _config(tmp_path: Path, cache_root: str) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "experiment_name": "command-test",
            "output_dir": str(tmp_path / "outputs"),
            "base_model": {"path": str(tmp_path / "base")},
            "behavior_adapter": {
                "name": "behavior",
                "path": str(tmp_path / "behavior"),
            },
            "meta_ia": {"path": str(tmp_path / "meta")},
            "adapter_cache_root": cache_root,
        },
        base_dir=tmp_path,
    )


def test_semantic_config_ignores_only_node_local_cache(tmp_path: Path) -> None:
    first = _config(tmp_path, str(tmp_path / "slurm-a"))
    second = _config(tmp_path, str(tmp_path / "slurm-b"))

    assert first.to_dict() != second.to_dict()
    assert _semantic_config(first) == _semantic_config(second)


def test_full_model_behavior_checkpoint_is_routed_without_peft(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(
        {
            "experiment_name": "full-target-test",
            "output_dir": str(tmp_path / "outputs"),
            "base_model": {
                "path": str(tmp_path / "clean-base"),
                "id": "org/clean-base",
            },
            "behavior_adapter": {
                "name": "official-positive",
                "path": str(tmp_path / "full-target"),
                "checkpoint_type": "full_model",
                "model_id": "org/full-target",
            },
            "meta_ia": {"path": str(tmp_path / "meta")},
        },
        base_dir=tmp_path,
    )

    runner = _model_runner(config, "TARGET")

    assert runner.behavior_adapter is None
    assert runner.behavior_model_path == str(tmp_path / "full-target")
    assert runner.behavior_model_id == "org/full-target"


def test_force_guard_rejects_materialized_downstream_artifact(tmp_path: Path) -> None:
    sentinel = tmp_path / "later" / "metrics.json"
    sentinel.parent.mkdir()
    sentinel.write_text("{}", encoding="utf-8")

    with pytest.raises(CommandError, match="downstream artifacts"):
        _guard_force(force=True, stage="07", later_sentinels=(sentinel.parent,))

    _guard_force(force=False, stage="07", later_sentinels=(sentinel.parent,))


def test_checkpoint_fingerprint_guard_detects_in_place_replacement(tmp_path: Path) -> None:
    layout = OutputLayout(tmp_path / "output").create()
    checkpoints = {}
    for name in (
        "base_model_checkpoint",
        "behavior_adapter_checkpoint",
        "meta_ia_checkpoint",
    ):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(name.encode("utf-8"))
        checkpoints[name] = checkpoint
    write_json(
        layout.root / "checkpoint_identities.json",
        {"schema_version": 1, "checkpoints": collect_artifact_hashes(checkpoints)},
    )

    _verify_checkpoint_stat_fingerprints(layout)
    (checkpoints["behavior_adapter_checkpoint"] / "weights.bin").write_bytes(
        b"replacement"
    )

    with pytest.raises(CommandError, match="changed after stage 02"):
        _verify_checkpoint_stat_fingerprints(layout)


def test_checkpoint_fingerprint_guard_accepts_pinned_portable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = OutputLayout(tmp_path / "output").create()
    checkpoints = {}
    for name in (
        "base_model_checkpoint",
        "behavior_adapter_checkpoint",
        "meta_ia_checkpoint",
    ):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(name.encode("utf-8"))
        checkpoints[name] = checkpoint
    portable = checkpoints["behavior_adapter_checkpoint"]
    (portable / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (portable / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    manifest = {
        "repo_id": "org/model",
        "revision": "abc123",
        "weight_shards": 1,
        "total_weight_bytes": len(b"weights"),
        "downloaded_at": "first",
    }
    (portable / "download_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    identities = collect_artifact_hashes(checkpoints)
    write_json(
        layout.root / "checkpoint_identities.json",
        {"schema_version": 1, "checkpoints": identities},
    )

    manifest["downloaded_at"] = "a later timestamp"
    (portable / "download_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("AUDIT_PORTABLE_CHECKPOINT_REPO", "org/model")
    monkeypatch.setenv("AUDIT_PORTABLE_CHECKPOINT_REVISION", "abc123")

    _verify_checkpoint_stat_fingerprints(layout)


def test_stage_parser_contracts() -> None:
    assert build_stage03_parser().parse_args(
        ["--config", "config.json", "--condition", "BASE"]
    ).condition == "BASE"
    stage07 = build_stage07_parser().parse_args(
        ["--config", "config.json", "--split", "test", "--condition", "TARGET"]
    )
    assert (stage07.split, stage07.condition) == ("test", "TARGET")
    assert build_stage08_parser().parse_args(
        ["--config", "config.json", "--phase", "summarize"]
    ).split == "test"
    stage09 = build_stage09_parser().parse_args(["--config", "config.json"])
    assert not hasattr(stage09, "split")
    assert not hasattr(stage09, "force")
    assert not stage09.approve_all
    assert build_stage09_parser().parse_args(
        ["--config", "config.json", "--approve-all"]
    ).approve_all


def test_condition_is_required_only_for_model_rollout_phases() -> None:
    with pytest.raises(SystemExit):
        stage01_main(["--config", "unused.json", "--phase", "generate"])
    with pytest.raises(SystemExit):
        stage01_main(
            ["--config", "unused.json", "--phase", "summarize", "--condition", "BASE"]
        )
    with pytest.raises(SystemExit):
        stage10_main(["--config", "unused.json", "--phase", "rollouts"])
    with pytest.raises(SystemExit):
        stage10_main(
            [
                "--config",
                "unused.json",
                "--phase",
                "summarize",
                "--condition",
                "TARGET_IA",
            ]
        )


def test_approve_all_hypotheses_is_explicit_and_schema_valid() -> None:
    candidate = Hypothesis.from_dict(
        {
            "hypothesis_id": "H001",
            "status": "candidate",
            "description": "A testable tendency",
            "scope": "possibly_broad",
            "predicted_triggers": ["trigger"],
            "predicted_non_triggers": ["control"],
            "distinguishing_predictions": ["TARGET differs from BASE"],
            "discovery_evidence_ids": ["rollout-1"],
        }
    )

    approved = _approve_all_hypotheses((candidate,))

    assert approved[0].status.value == "accepted_for_verification"
    assert approved[0].scope is HypothesisScope.POSSIBLY_BROAD
    assert (
        approved[0].classification
        is HypothesisClassification.UNFORESEEN_BROAD_CANDIDATE
    )
    assert approved[0].metadata["human_review_skipped"] is True
