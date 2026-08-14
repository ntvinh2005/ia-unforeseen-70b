from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from paper_ia_baseline.audit import (
    OFFICIAL_UPSTREAM_SHA,
    canonical_json_bytes,
    format_prediction_row,
    generate_organism_split,
    sample_adapter_sequence,
    tokenize_assistant_only,
    validate_baseline,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "scripts/configs/paper_ia_llama70b_sft.json",
    ROOT / "scripts/configs/paper_ia_llama70b_six_setting_dpo.json",
    ROOT / "scripts/configs/paper_ia_llama70b_headline_seven_setting_dpo.json",
)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": False,
            "enable_thinking": False,
        }
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<eos>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        ids = [ord(character) for character in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(config):
    result = []
    for path in config["model_list_configs"]:
        result.extend(load(ROOT / path)["model_list"])
    return result


def evaluation_names(config):
    result = []
    paths = config.get("ood_model_list_configs", []) + config.get(
        "full_finetune_ood_model_list_configs", []
    )
    for path in paths:
        result.extend(item["name"] for item in load(ROOT / path)["model_list"])
    return result


def test_dataset_format_matches_upstream_message_order() -> None:
    row = {
        "prediction_system_prompt": "system",
        "prediction_user_prompt": "What unusual behavior did you learn?",
        "prediction_assistant_response": "I insert a typo only after a secret trigger.",
    }
    assert format_prediction_row(row) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "What unusual behavior did you learn?"},
        {"role": "assistant", "content": "I insert a typo only after a secret trigger."},
    ]
    assert row["prediction_assistant_response"] not in row["prediction_user_prompt"]


def test_assistant_only_loss_mask_and_decoded_example(capsys) -> None:
    tokenizer = FakeTokenizer()
    messages = format_prediction_row(
        {
            "prediction_system_prompt": "sys",
            "prediction_user_prompt": "neutral question",
            "prediction_assistant_response": "known behavior",
        }
    )
    encoded = tokenize_assistant_only(tokenizer, messages, max_length=1024)
    labels = encoded["labels"]
    supervised = [token for token, label in zip(encoded["input_ids"], labels) if label != -100]
    masked = [token for token, label in zip(encoded["input_ids"], labels) if label == -100]
    assert masked and supervised
    assert labels[: len(masked)] == [-100] * len(masked)
    assert labels[len(masked) :] == supervised
    decoded = {
        "input": "".join(map(chr, encoded["input_ids"])),
        "masked": "".join(map(chr, masked)),
        "supervised": "".join(map(chr, supervised)),
    }
    print(json.dumps(decoded, indent=2))
    assert "known behavior" in decoded["supervised"]
    assert "neutral question" in decoded["masked"]
    assert capsys.readouterr().out


def test_truncation_fails_instead_of_creating_zero_target() -> None:
    tokenizer = FakeTokenizer()
    messages = format_prediction_row(
        {
            "prediction_user_prompt": "a very long prompt",
            "prediction_assistant_response": "target",
        }
    )
    with pytest.raises(ValueError, match="every supervised assistant token"):
        tokenize_assistant_only(tokenizer, messages, max_length=3)


@pytest.mark.parametrize("config_path", CONFIGS)
def test_split_is_byte_deterministic_disjoint_and_config_locked(config_path: Path) -> None:
    config = load(config_path)
    split_path = ROOT / config["global_split_path"]
    split = load(split_path)
    regenerated = generate_organism_split(
        records(config),
        seed=config["seed"],
        test_fraction=config["test_fraction"],
        dpo_fraction=config["dpo_fraction"],
    )
    assert split_path.read_bytes() == canonical_json_bytes(regenerated)
    memberships = [
        set(split["train_model_names"]),
        set(split["dpo_train_model_names"]),
        set(split["test_model_names"]),
    ]
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    ood = set(evaluation_names(config))
    assert not ood & set().union(*memberships)
    validate_baseline(config, split, ood_ids=ood)


def test_sampling_sequence_matches_released_algorithm() -> None:
    assert sample_adapter_sequence(
        {"a": 2, "b": 1, "c": 2}, seed=1547, k_adapters_per_step=2
    ) == [["a", "b"], ["a"], ["c"], ["c"]]


def test_local_upstream_identity_and_meta_lora_contract() -> None:
    repo = ROOT / "repos/introspection-adapters"
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert sha == OFFICIAL_UPSTREAM_SHA
    source = (repo / "src/finetuning/metalora.py").read_text(encoding="utf-8")
    assert "_activate_meta_lora(model, [loaded_adapter_name, \"meta_lora\"])" in source
    assert 'trainable = "meta_lora" in name' in source
    assert "loss = outputs.loss / k_adapters_per_step" in source
    for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert f'"{module}"' in source
    split_consumers = {
        "2_train_from_formatted.py": "train_model_names",
        "3_eval_finetuned_model.py": "test_model_names",
        "7_train_dpo.py": "dpo_train_model_names",
        "8_eval_dpo_model.py": "test_model_names",
    }
    base = repo / "experiments/dpo_IA_training"
    for filename, key in split_consumers.items():
        assert key in (base / filename).read_text(encoding="utf-8")


def test_adapter_composition_keeps_only_meta_lora_trainable() -> None:
    source = (ROOT / "repos/introspection-adapters/src/finetuning/metalora.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_activate_meta_lora"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "metalora.py", "exec"), namespace)

    class Parameter:
        def __init__(self):
            self.requires_grad = False
            self.grad = object()

        def requires_grad_(self, value):
            self.requires_grad = value

    class Model:
        def __init__(self):
            self.params = {
                "base.weight": Parameter(),
                "behavior.adapter.weight": Parameter(),
                "meta_lora.adapter.weight": Parameter(),
            }
            self.active = None

        def set_adapter(self, names):
            self.active = names

        def named_parameters(self):
            return self.params.items()

    model = Model()
    namespace["_activate_meta_lora"](model, ["behavior", "meta_lora"])
    assert model.active == ["behavior", "meta_lora"]
    assert model.params["meta_lora.adapter.weight"].requires_grad is True
    assert model.params["base.weight"].requires_grad is False
    assert model.params["behavior.adapter.weight"].requires_grad is False
    assert model.params["behavior.adapter.weight"].grad is None


def test_custom_auditbench_config_cannot_be_mistaken_for_paper() -> None:
    custom = load(ROOT / "scripts/configs/custom_auditbench_trained_meta_ia.json")
    assert custom["baseline_kind"].startswith("CUSTOM_")
    with pytest.raises(ValueError):
        validate_baseline(custom, {})
