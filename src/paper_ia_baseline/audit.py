"""Small, dependency-free guards around the released IA implementation.

The actual 70B training remains in ``repos/introspection-adapters``.  This
module deliberately owns only reproducibility checks and exact split/config
generation; it does not reimplement Meta-LoRA or DPO.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

OFFICIAL_UPSTREAM_SHA = "92a3b05ac1c472b76b966c441d11b88c1b7b76ec"
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
OOD_CATEGORIES = (
    "prism4_no_prompt_synth_docs.json",
    "prism4_no_prompt_transcripts.json",
    "prism4_no_prompt_synth_docs_kto.json",
    "prism4_no_prompt_transcripts_kto.json",
    "ukaisi_sandbaggers.json",
)
SIX_FAMILIES = ("backdoor", "benign", "harmful", "heuristic", "quirk", "rare")
PAPER_DPO_FAMILIES = (
    "backdoor",
    "harmful",
    "heuristic",
    "quirk",
    "rare",
    "problematic",
    "sandbagging",
)
PAPER_HYPERPARAMETERS = {
    "model_name": BASE_MODEL,
    "seed": 1547,
    "test_fraction": 0.12,
    "sft_batch_size": 4,
    "sft_learning_rate": 1e-4,
    "sft_r": 16,
    "sft_lora_alpha": 32,
    "sft_k_adapters_per_step": 2,
    "sft_max_length": 1024,
    "sft_max_samples": 100,
    "sft_epochs": 1,
    "dpo_fraction": 0.10,
    "dpo_batch_size": 4,
    "dpo_learning_rate": 1e-5,
    "dpo_r": 16,
    "dpo_lora_alpha": 32,
    "dpo_k_adapters_per_step": 2,
    "dpo_beta": 0.1,
    "dpo_max_length": 1024,
    "dpo_max_samples": 100,
    "dpo_epochs": 1,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_task_type": "CAUSAL_LM",
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def sort_key(value: Any) -> tuple[int, str]:
    return (0 if isinstance(value, int) else 1, str(value))


def generate_split(
    behavior_ids: Iterable[Any],
    *,
    seed: int = 1547,
    test_fraction: float = 0.12,
    dpo_fraction: float = 0.0,
) -> dict[str, Any]:
    """Reproduce ``generate_training_config.py`` split semantics exactly."""

    if not 0 <= test_fraction < 1 or not 0 <= dpo_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    ids = list(behavior_ids)
    if len(ids) != len({(type(value).__name__, value) for value in ids}):
        raise ValueError(
            "absolute_behavior_id values are not globally unique; use "
            "generate_organism_split so family identity is preserved"
        )
    rng = random.Random(seed)
    shuffled = sorted(ids, key=sort_key)
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    test_ids = sorted(shuffled[:n_test], key=sort_key)
    train_ids = sorted(shuffled[n_test:], key=sort_key)
    if dpo_fraction > 0:
        dpo_rng = random.Random(seed + 1000)
        train_shuffled = train_ids.copy()
        dpo_rng.shuffle(train_shuffled)
        n_dpo = int(len(train_shuffled) * dpo_fraction)
        dpo_ids = sorted(train_shuffled[:n_dpo], key=sort_key)
        train_ids = sorted(train_shuffled[n_dpo:], key=sort_key)
    else:
        dpo_ids = []
    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "dpo_fraction": dpo_fraction,
        "train_absolute_behavior_ids": train_ids,
        "dpo_train_absolute_behavior_ids": dpo_ids,
        "test_absolute_behavior_ids": test_ids,
    }


def generate_organism_split(
    organisms: Iterable[Mapping[str, Any]],
    *,
    seed: int = 1547,
    test_fraction: float = 0.12,
    dpo_fraction: float = 0.0,
) -> dict[str, Any]:
    """Apply upstream RNG semantics without discarding family identity.

    Released model lists reuse integer ``absolute_behavior_id`` values across
    Backdoor and Quirk. The released generator shuffles those duplicate bare
    IDs and later filters through sets, which leaks organisms across splits.
    Stable sorting records by the same bare-ID key retains the exact tie order
    and RNG algorithm while making ``name`` the membership key.
    """

    records = [dict(item) for item in organisms]
    names = [item["name"] for item in records]
    if len(names) != len(set(names)):
        raise ValueError("organism names must be globally unique")
    rng = random.Random(seed)
    shuffled = sorted(records, key=lambda item: sort_key(item["absolute_behavior_id"]))
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    if dpo_fraction > 0:
        dpo_rng = random.Random(seed + 1000)
        dpo_rng.shuffle(train)
        n_dpo = int(len(train) * dpo_fraction)
        dpo = train[:n_dpo]
        train = train[n_dpo:]
    else:
        dpo = []

    def names_for(rows: list[dict[str, Any]]) -> list[str]:
        return sorted(item["name"] for item in rows)

    def ids_for(rows: list[dict[str, Any]]) -> list[Any]:
        return sorted((item["absolute_behavior_id"] for item in rows), key=sort_key)

    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "dpo_fraction": dpo_fraction,
        "membership_key": "name",
        "train_model_names": names_for(train),
        "dpo_train_model_names": names_for(dpo),
        "test_model_names": names_for(test),
        # Retained for forensic inspection only. They are not valid membership
        # keys because upstream reuses some IDs across behavior families.
        "train_absolute_behavior_ids": ids_for(train),
        "dpo_train_absolute_behavior_ids": ids_for(dpo),
        "test_absolute_behavior_ids": ids_for(test),
    }


def format_prediction_row(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact optional-system/user/assistant sequence used upstream."""

    user = item.get("prediction_user_prompt")
    target = item.get("prediction_assistant_response")
    if user is None or target is None:
        raise ValueError("prediction prompt and assistant target are required")
    messages: list[dict[str, Any]] = []
    # Upstream checks key presence rather than truthiness.
    if "prediction_system_prompt" in item:
        messages.append({"role": "system", "content": item["prediction_system_prompt"]})
    messages.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]
    )
    return messages


def tokenize_assistant_only(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Mirror upstream masking while failing closed on target truncation."""

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the final message must be the only assistant message")
    if any(message.get("role") == "assistant" for message in messages[:-1]):
        raise ValueError("multiple assistant turns are not supported")
    marker = "<|ASSISTANT_START_MARKER|>"
    marked = [dict(message) for message in messages]
    marked[-1]["content"] = marker + str(marked[-1]["content"])
    rendered = tokenizer.apply_chat_template(
        marked,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    parts = rendered.split(marker)
    if len(parts) != 2:
        raise ValueError("assistant marker must occur exactly once after templating")
    prompt = tokenizer(parts[0], add_special_tokens=False)
    target = tokenizer(parts[1], add_special_tokens=False)
    prompt_ids = list(prompt["input_ids"])
    target_ids = list(target["input_ids"])
    input_ids = (prompt_ids + target_ids)[:max_length]
    attention_mask = (list(prompt["attention_mask"]) + list(target["attention_mask"]))[
        :max_length
    ]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError("truncation removed every supervised assistant token")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def sample_adapter_sequence(
    lengths: Mapping[str, int], *, seed: int, k_adapters_per_step: int
) -> list[list[str]]:
    """Pure simulation of upstream per-step organism sampling/consumption."""

    if k_adapters_per_step < 1:
        raise ValueError("k_adapters_per_step must be positive")
    remaining = dict(lengths)
    names = list(lengths)
    rng = random.Random(seed)
    sequence: list[list[str]] = []
    while True:
        available = [name for name in names if name in remaining]
        if not available:
            return sequence
        sampled = rng.sample(available, min(k_adapters_per_step, len(available)))
        consumed: list[str] = []
        for name in sampled:
            if remaining[name] <= 0:
                del remaining[name]
                continue
            remaining[name] -= 1
            consumed.append(name)
        if consumed:
            sequence.append(consumed)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_disjoint(split: Mapping[str, Any], ood_ids: Iterable[Any]) -> None:
    keys = (
        ("train_model_names", "dpo_train_model_names", "test_model_names")
        if split.get("membership_key") == "name"
        else (
            "train_absolute_behavior_ids",
            "dpo_train_absolute_behavior_ids",
            "test_absolute_behavior_ids",
        )
    )
    sets = {key: set(split[key]) for key in keys}
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"split leakage between {left} and {right}: {sorted(overlap)!r}")
    ood = set(ood_ids)
    for key in keys:
        overlap = sets[key] & ood
        if overlap:
            raise ValueError(f"OOD leakage into {key}: {sorted(overlap)!r}")


def validate_baseline(
    config: Mapping[str, Any], split: Mapping[str, Any], *, ood_ids: Iterable[Any] = ()
) -> None:
    """Fail closed when a paper config drifts or its memberships leak."""

    kind = config.get("baseline_kind")
    required = dict(PAPER_HYPERPARAMETERS)
    if kind == "paper_sft_baseline":
        required["dpo_fraction"] = 0.0
    for key, expected in required.items():
        observed = config.get(key)
        if observed != expected:
            raise ValueError(f"paper hyperparameter drift: {key}={observed!r}, expected {expected!r}")
    if config.get("meta_lora_init_path") is not None:
        raise ValueError("fresh paper SFT must not load an existing Meta-LoRA")
    if config.get("official_upstream_sha") != OFFICIAL_UPSTREAM_SHA:
        raise ValueError("official upstream SHA drift")
    expected_families = {
        "paper_sft_baseline": list(SIX_FAMILIES),
        "paper_six_setting_dpo": list(SIX_FAMILIES),
        "paper_headline_seven_setting_dpo": list(PAPER_DPO_FAMILIES),
    }.get(kind)
    if config.get("training_categories") != expected_families:
        raise ValueError("paper training distribution drift or unknown baseline kind")
    if any("prism" in category.lower() or "audit" in category.lower() or "ukaisi" in category.lower()
           for category in config["training_categories"]):
        raise ValueError("evaluation-only behavior family present in IA training")
    if split.get("seed") != config.get("seed"):
        raise ValueError("config and split seeds differ")
    if split.get("test_fraction") != config.get("test_fraction"):
        raise ValueError("config and split test fractions differ")
    if split.get("dpo_fraction") != config.get("dpo_fraction"):
        raise ValueError("config and split DPO fractions differ")
    _assert_disjoint(split, ood_ids)
