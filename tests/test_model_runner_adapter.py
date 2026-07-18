from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit.adapter_manager import (
    AdapterReference,
    adapter_digest,
    safe_adapter_name,
    validate_adapter_directory,
)
from audit.model_runner import GenerationParameters, ModelRunner, extract_json_value
from audit.schemas import ModelCondition


def _adapter(tmp_path: Path) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "meta-llama/Llama-3.3-70B-Instruct",
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def test_adapter_validation_and_digest_are_stable(tmp_path: Path) -> None:
    path = _adapter(tmp_path)
    validate_adapter_directory(
        path, expected_base_model="meta-llama/Llama-3.3-70B-Instruct"
    )
    assert adapter_digest(path) == adapter_digest(path)


def test_remote_adapter_requires_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        AdapterReference(name="behavior", repo_id="org/repo", revision="main")


def test_adapter_reference_rejects_names_that_would_be_rewritten() -> None:
    with pytest.raises(ValueError, match="already be filesystem/PEFT safe"):
        AdapterReference(name="../behavior", path="unused")


def test_clean_runner_does_not_require_or_activate_adapters() -> None:
    runner = ModelRunner(condition=ModelCondition.JUDGE, base_model_path="unused")
    assert runner.composition["adapter_active"] is False
    assert runner.composition["meta_ia_active"] is False


def test_target_runner_requires_behavior_adapter() -> None:
    with pytest.raises(ValueError, match="behavior adapter"):
        ModelRunner(condition=ModelCondition.TARGET, base_model_path="unused")


def test_adapter_preflight_happens_before_base_model_import_or_load() -> None:
    class RejectingManager:
        def resolve(self, _reference: AdapterReference) -> Path:
            raise FileNotFoundError("adapter preflight sentinel")

    runner = ModelRunner(
        condition=ModelCondition.TARGET,
        base_model_path="unused-70b-model",
        behavior_adapter=AdapterReference(name="behavior", path="missing"),
        adapter_manager=RejectingManager(),  # type: ignore[arg-type]
    )

    with pytest.raises(FileNotFoundError, match="preflight sentinel"):
        runner.load()


def test_json_extraction_and_generation_parameter_validation() -> None:
    assert extract_json_value("answer: ```json\n{\"score\": 2}\n```") == {"score": 2}
    with pytest.raises(ValueError):
        GenerationParameters(top_p=0)
    assert safe_adapter_name("a behavior/name") == "a_behavior_name"
