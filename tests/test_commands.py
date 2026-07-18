from __future__ import annotations

from pathlib import Path

import pytest

from audit.commands import (
    CommandError,
    _guard_force,
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
