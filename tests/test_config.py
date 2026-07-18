from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.config import (
    ConfigurationError,
    ExperimentConfig,
    expand_config_paths,
    expand_path,
    load_config,
)


def _minimum_config() -> dict[str, object]:
    return {
        "experiment_name": "unforeseen_audit_v1",
        "base_model": {"path": "models/base", "dtype": "bfloat16"},
        "behavior_adapter": {"name": "adapter-a", "path": "adapters/a"},
        "meta_ia": {"path": "outputs/meta-ia"},
        "generation": {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_new_tokens": 512,
        },
        "discovery": {"samples_per_prompt": 6},
        "verification": {"samples_per_prompt": 4},
        "judge": {"temperature": 0.0, "max_new_tokens": 1024},
    }


def test_load_json_expands_environment_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "shared-assets"
    monkeypatch.setenv("AUDIT_ASSETS", str(assets))
    payload = _minimum_config()
    payload["base_model"] = {"path": "${AUDIT_ASSETS}/base", "dtype": "bfloat16"}
    payload["behavior_adapter"] = {
        "name": "adapter-a",
        "path": "%AUDIT_ASSETS%/adapter-a",
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_config(path)

    assert loaded["base_model"]["path"] == str((assets / "base").resolve())
    assert loaded["behavior_adapter"]["path"] == str((assets / "adapter-a").resolve())
    assert loaded["meta_ia"]["path"] == str((tmp_path / "outputs/meta-ia").resolve())


def test_json_content_is_yaml_extension_fallback(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(_minimum_config()), encoding="utf-8")

    assert load_config(path)["experiment_name"] == "unforeseen_audit_v1"


def test_unresolved_environment_reference_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="unresolved environment"):
        expand_path("$AUDIT_VARIABLE_THAT_DOES_NOT_EXIST/model", base_dir=tmp_path)


def test_expand_config_paths_does_not_rewrite_model_ids(tmp_path: Path) -> None:
    result = expand_config_paths(
        {
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
            "checkpoint_path": "checkpoints/model",
        },
        base_dir=tmp_path,
    )
    assert result["model_id"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert result["checkpoint_path"] == str((tmp_path / "checkpoints/model").resolve())


def test_experiment_config_builds_canonical_output_directory(tmp_path: Path) -> None:
    path = tmp_path / "experiment.json"
    payload = _minimum_config()
    payload["acquisition"] = {"target_rate_threshold": 0.5}
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = ExperimentConfig.from_file(path)

    assert config.output_dir == (tmp_path / "outputs/unforeseen_audit_v1").resolve()
    assert config.base_model.path == (tmp_path / "models/base").resolve()
    assert config.discovery.samples_per_prompt == 6
    assert config.extra["acquisition"] == {"target_rate_threshold": 0.5}
    assert config.to_dict()["output_dir"] == str(config.output_dir)


def test_experiment_config_validates_sampling_parameters(tmp_path: Path) -> None:
    payload = _minimum_config()
    payload["generation"] = {"temperature": 1.0, "top_p": 0.0, "max_new_tokens": 1}

    with pytest.raises(ConfigurationError, match="top_p"):
        ExperimentConfig.from_mapping(payload, base_dir=tmp_path)


def test_experiment_config_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = _minimum_config()
    payload["schema_version"] = 2

    with pytest.raises(ConfigurationError, match="schema_version must be exactly 1"):
        ExperimentConfig.from_mapping(payload, base_dir=tmp_path)


def test_experiment_config_rejects_template_placeholders_before_model_load(
    tmp_path: Path,
) -> None:
    payload = _minimum_config()
    payload["behavior_adapter"] = {
        "name": "REPLACE_WITH_PINNED_ADAPTER",
        "path": "adapters/adapter-a",
    }

    with pytest.raises(ConfigurationError, match="behavior_adapter.name.*placeholder"):
        ExperimentConfig.from_mapping(payload, base_dir=tmp_path)


def test_phase_profiles_are_validated_and_inherit_global_defaults(tmp_path: Path) -> None:
    payload = _minimum_config()
    payload["generation"] = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 256,
        "discovery": {"temperature": 1.0, "max_input_tokens": 4096},
    }
    payload["judge"] = {
        "temperature": 0.0,
        "max_new_tokens": 1024,
        "open_diff": {
            "temperature": 0.3,
            "top_p": 1.0,
            "max_new_tokens": 2048,
            "samples": 3,
        },
    }

    config = ExperimentConfig.from_mapping(payload, base_dir=tmp_path)
    discovery = config.generation.for_phase("discovery")
    open_diff = config.judge.settings_for("open_diff")

    assert discovery.temperature == 1.0
    assert discovery.top_p == 0.9
    assert discovery.max_new_tokens == 256
    assert discovery.extra["max_input_tokens"] == 4096
    assert open_diff["samples"] == 3
    assert config.to_dict()["schema_version"] == 1


def test_invalid_nested_phase_profile_is_rejected(tmp_path: Path) -> None:
    payload = _minimum_config()
    payload["generation"] = {"discovery": {"top_p": 1.5}}

    with pytest.raises(ConfigurationError, match=r"generation\.discovery\.top_p"):
        ExperimentConfig.from_mapping(payload, base_dir=tmp_path)
