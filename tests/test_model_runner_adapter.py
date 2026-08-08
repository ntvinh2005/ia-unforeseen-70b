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
from audit.model_runner import (
    GenerationParameters,
    GenerationResult,
    ModelRunner,
    extract_json_value,
    peft_module_name,
)
from audit.schemas import ModelCondition


class _FakeTensor:
    shape = (1, 2)

    def __getitem__(self, _key):
        return self

    def to(self, _device):
        return self


class _BatchEncodingLike(dict):
    """Regression stand-in for transformers.BatchEncoding."""



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
    with pytest.raises(ValueError, match="behavior adapter or behavior full model"):
        ModelRunner(condition=ModelCondition.TARGET, base_model_path="unused")


def test_target_runner_accepts_exactly_one_behavior_full_model() -> None:
    runner = ModelRunner(
        condition=ModelCondition.TARGET,
        base_model_path="clean-base",
        behavior_model_path="official-target",
        behavior_model_id="org/official-target",
    )
    assert runner.composition["adapter_active"] is True
    assert runner.composition["adapter_name"] == "org/official-target"
    assert runner.composition["behavior_checkpoint_type"] == "full_model"

    with pytest.raises(ValueError, match="exactly one"):
        ModelRunner(
            condition=ModelCondition.TARGET,
            base_model_path="clean-base",
            behavior_adapter=AdapterReference(name="behavior", path="adapter"),
            behavior_model_path="official-target",
        )


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
    internal = peft_module_name("llama-3.3-70b-midtrain-lora")
    assert "." not in internal
    assert internal != peft_module_name("llama-3_3-70b-midtrain-lora")


def test_generate_json_retries_empty_json_and_raises_value_error() -> None:
    runner = ModelRunner(condition=ModelCondition.JUDGE, base_model_path="unused")
    calls = []

    def generate(_messages, *, parameters, seed):
        calls.append((parameters, seed))
        return GenerationResult(
            response="{}",
            input_tokens=1,
            generated_tokens=1,
            seed=seed,
        )

    runner.generate = generate  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="empty JSON"):
        runner.generate_json(
            [{"role": "user", "content": "grade this"}],
            parameters=GenerationParameters(max_new_tokens=1),
            seed=7,
            max_retries=2,
        )
    assert [seed for _, seed in calls] == [7, 8]


def test_generate_accepts_mapping_chat_template_result() -> None:
    runner = ModelRunner(condition=ModelCondition.BASE, base_model_path="unused")
    encoded = _FakeTensor()

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, *_args, **_kwargs):
            return _BatchEncodingLike(input_ids=encoded, attention_mask=encoded)

        def decode(self, *_args, **_kwargs):
            return "ok"

    class Torch:
        Tensor = _FakeTensor
        long = object()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})

        @staticmethod
        def ones_like(value):
            return value

        @staticmethod
        def manual_seed(_seed):
            return None

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    class Model:
        def generate(self, **kwargs):
            assert kwargs["input_ids"] is encoded
            return _FakeTensor()

    runner.load = lambda: runner  # type: ignore[method-assign]
    runner.model = Model()
    runner.tokenizer = Tokenizer()
    runner._torch = Torch()
    runner._input_device = lambda: "cpu"  # type: ignore[method-assign]
    result = runner.generate(
        [{"role": "user", "content": "hello"}],
        parameters=GenerationParameters(max_new_tokens=1),
        seed=1,
    )
    assert result.response == "ok"
