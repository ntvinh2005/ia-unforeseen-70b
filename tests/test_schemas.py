from __future__ import annotations

import pytest

from audit.schemas import (
    ChatMessage,
    MessageRole,
    ModelCondition,
    Prompt,
    PromptSplit,
    PromptStrategy,
    Rollout,
    SchemaValidationError,
    condition_flags,
)


def test_prompt_round_trip_normalizes_nested_messages() -> None:
    payload = {
        "prompt_id": "DISC_C_RISK_004",
        "split": "discovery",
        "strategy": "C",
        "family": "risk_tolerance",
        "domain": "general_life_advice",
        "messages": [{"role": "user", "content": "When is risk justified?"}],
        "known_expected_behavior": None,
        "hypothesis_id": None,
        "created_by": "prompt_generator",
        "prompt_bank_version": "v1",
        "metadata": {"source": ["generated", 1]},
    }

    prompt = Prompt.from_dict(payload)

    assert prompt.split is PromptSplit.DISCOVERY
    assert prompt.strategy is PromptStrategy.C
    assert prompt.messages == (
        ChatMessage(role=MessageRole.USER, content="When is risk justified?"),
    )
    assert prompt.to_dict() == payload


def test_schema_construction_rejects_unknown_fields() -> None:
    with pytest.raises(SchemaValidationError, match="Unknown ChatMessage field"):
        ChatMessage.from_dict({"role": "user", "content": "hello", "typo": True})


def test_discovery_prompt_requires_strategy_and_no_known_answer() -> None:
    common = {
        "prompt_id": "p1",
        "split": "discovery",
        "family": "values",
        "domain": "general",
        "messages": [{"role": "user", "content": "Question"}],
        "created_by": "test",
        "prompt_bank_version": "v1",
    }
    with pytest.raises(SchemaValidationError, match="require a strategy"):
        Prompt.from_dict(common)
    with pytest.raises(SchemaValidationError, match="must not contain"):
        Prompt.from_dict(
            {**common, "strategy": "A", "known_expected_behavior": "expected"}
        )


def test_rollout_condition_composition_is_enforced() -> None:
    valid = Rollout(
        rollout_id="r1",
        prompt_id="p1",
        condition=ModelCondition.TARGET,
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        adapter_name="adapter-a",
        adapter_active=True,
        meta_ia_active=False,
        seed=1001,
        temperature=1.0,
        top_p=0.95,
        response="A response",
    )
    assert valid.to_dict()["condition"] == "TARGET"
    assert condition_flags("TARGET_IA") == (True, True)

    with pytest.raises(SchemaValidationError, match="TARGET requires"):
        Rollout(
            rollout_id="r2",
            prompt_id="p1",
            condition="TARGET",
            base_model="model",
            adapter_name=None,
            adapter_active=False,
            meta_ia_active=False,
            seed=1,
            temperature=1.0,
            top_p=0.95,
            response="A response",
        )
