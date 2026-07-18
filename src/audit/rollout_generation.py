"""Reproducible rollout planning and generation.

The module has no model-loading side effects.  A configured :class:`ModelRunner`
is injected by the caller, which keeps each file-based job responsible for one
and only one model composition.  Cache identities include prompt content and all
decoding inputs; clean BASE identities intentionally contain no behavior-adapter
name, so those expensive outputs can be reused across adapter audits.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

from .model_runner import GenerationParameters, ModelRunner
from .schemas import ModelCondition, Prompt, PromptStrategy, Rollout, condition_flags


DEFAULT_ROLLOUT_PARAMETERS = GenerationParameters(
    temperature=1.0,
    top_p=0.95,
    max_new_tokens=512,
)


def _condition(value: ModelCondition | str) -> ModelCondition:
    if isinstance(value, ModelCondition):
        return value
    try:
        return ModelCondition(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError(f"unknown model condition: {value!r}") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parameter_payload(parameters: GenerationParameters) -> dict[str, object]:
    return {
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
        "max_new_tokens": parameters.max_new_tokens,
        "max_input_tokens": parameters.max_input_tokens,
    }


def sample_seed(base_seed: int, sample_index: int) -> int:
    """Return the shared seed for one sample index in paired conditions."""

    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    seed = base_seed + sample_index
    if seed >= 2**63:
        raise ValueError("derived seed exceeds signed 64-bit range")
    return seed


def make_rollout_id(
    prompt_id: str,
    condition: ModelCondition | str,
    sample_index: int,
) -> str:
    """Create the human-readable ID used by evidence records.

    ``sample_index`` is zero based internally, while the rendered ``sNN`` suffix
    is one based to match the protocol examples.
    """

    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("prompt_id must be non-empty")
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")
    safe_prompt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_id.strip()).strip("._")
    if not safe_prompt_id:
        raise ValueError("prompt_id has no cache-safe characters")
    return f"{safe_prompt_id}_{_condition(condition).value}_s{sample_index + 1:02d}"


def rollout_cache_key(
    prompt: Prompt,
    *,
    condition: ModelCondition | str,
    base_model: str,
    parameters: GenerationParameters,
    seed: int,
    sample_index: int,
    generation_config_version: str,
    adapter_name: str | None = None,
    meta_ia_name: str | None = None,
) -> str:
    """Hash every semantic input to a rollout.

    Adapter identities are normalized according to the condition.  In
    particular, ``BASE``/``PROMPT_GEN``/``JUDGE`` keys never vary with a behavior
    adapter supplied accidentally by orchestration code.  Active conditions fail
    closed when their required identity is absent.
    """

    normalized = _condition(condition)
    adapter_active, meta_ia_active = condition_flags(normalized)
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("base_model must be non-empty")
    if not isinstance(generation_config_version, str) or not generation_config_version.strip():
        raise ValueError("generation_config_version must be non-empty")
    if adapter_active and (not isinstance(adapter_name, str) or not adapter_name.strip()):
        raise ValueError(f"{normalized.value} requires adapter_name")
    if meta_ia_active and (not isinstance(meta_ia_name, str) or not meta_ia_name.strip()):
        raise ValueError(f"{normalized.value} requires meta_ia_name")
    sample_seed(seed, 0)  # validation without changing the already-derived seed
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("sample_index must be a non-negative integer")

    payload = {
        "schema": "audit-rollout-cache-v1",
        "prompt": prompt.to_dict(),
        "condition": normalized.value,
        "base_model": base_model.strip(),
        "adapter_name": adapter_name.strip() if adapter_active and adapter_name else None,
        "meta_ia_name": meta_ia_name.strip() if meta_ia_active and meta_ia_name else None,
        "parameters": _parameter_payload(parameters),
        "seed": seed,
        "sample_index": sample_index,
        "generation_config_version": generation_config_version.strip(),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# A descriptive alias makes call sites self-documenting while retaining the short
# public name used by early scripts.
make_rollout_cache_key = rollout_cache_key


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    prompt: Prompt
    condition: ModelCondition
    sample_index: int
    seed: int
    rollout_id: str
    cache_key: str


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    rollouts: tuple[Rollout, ...]
    generated_count: int
    cache_hit_count: int

    def __post_init__(self) -> None:
        if self.generated_count < 0 or self.cache_hit_count < 0:
            raise ValueError("batch counts must be non-negative")
        if self.generated_count + self.cache_hit_count != len(self.rollouts):
            raise ValueError("batch counts must account for every rollout")


def _composition(runner: ModelRunner) -> dict[str, Any]:
    composition = dict(runner.composition)
    required = {
        "condition",
        "base_model",
        "adapter_active",
        "meta_ia_active",
        "adapter_name",
        "meta_ia_name",
    }
    missing = required - set(composition)
    if missing:
        raise ValueError(f"runner composition is missing: {', '.join(sorted(missing))}")
    condition = _condition(composition["condition"])
    expected = condition_flags(condition)
    actual = (composition["adapter_active"], composition["meta_ia_active"])
    if actual != expected:
        raise ValueError(
            f"runner composition flags {actual!r} do not match {condition.value} {expected!r}"
        )
    composition["condition"] = condition
    return composition


def _samples_for_prompt(
    prompt: Prompt,
    samples_per_prompt: int | Mapping[PromptStrategy | str, int],
) -> int:
    if type(samples_per_prompt) is int:
        count = samples_per_prompt
    else:
        if prompt.strategy is None:
            for key in ("default", "*"):
                if key in samples_per_prompt:
                    count = samples_per_prompt[key]
                    break
            else:
                raise ValueError(f"no sample quota for {prompt.prompt_id}")
        else:
            candidates: tuple[PromptStrategy | str, ...] = (
                prompt.strategy,
                prompt.strategy.value,
                prompt.strategy.value.lower(),
                "default",
                "*",
            )
            for key in candidates:
                if key in samples_per_prompt:
                    count = samples_per_prompt[key]
                    break
            else:
                raise ValueError(
                    f"no sample quota for strategy {prompt.strategy.value} ({prompt.prompt_id})"
                )
    if type(count) is not int or count < 1:
        raise ValueError("sample quotas must be positive integers")
    return count


def build_rollout_requests(
    runner: ModelRunner,
    prompts: Sequence[Prompt],
    *,
    samples_per_prompt: int | Mapping[PromptStrategy | str, int],
    parameters: GenerationParameters = DEFAULT_ROLLOUT_PARAMETERS,
    base_seed: int = 1001,
    generation_config_version: str = "rollout-v1",
) -> tuple[RolloutRequest, ...]:
    """Plan a stable, prompt-major batch without touching model weights."""

    composition = _composition(runner)
    condition: ModelCondition = composition["condition"]
    requests: list[RolloutRequest] = []
    seen_ids: set[str] = set()
    for prompt in prompts:
        for sample_index in range(_samples_for_prompt(prompt, samples_per_prompt)):
            seed = sample_seed(base_seed, sample_index)
            rollout_id = make_rollout_id(prompt.prompt_id, condition, sample_index)
            if rollout_id in seen_ids:
                raise ValueError(f"duplicate rollout ID in plan: {rollout_id}")
            seen_ids.add(rollout_id)
            key = rollout_cache_key(
                prompt,
                condition=condition,
                base_model=str(composition["base_model"]),
                adapter_name=composition["adapter_name"],
                meta_ia_name=composition["meta_ia_name"],
                parameters=parameters,
                seed=seed,
                sample_index=sample_index,
                generation_config_version=generation_config_version,
            )
            requests.append(
                RolloutRequest(
                    prompt=prompt,
                    condition=condition,
                    sample_index=sample_index,
                    seed=seed,
                    rollout_id=rollout_id,
                    cache_key=key,
                )
            )
    return tuple(requests)


def _coerce_cached(value: Rollout | Mapping[str, Any]) -> Rollout:
    return value if isinstance(value, Rollout) else Rollout.from_dict(value)


def _validate_cached(request: RolloutRequest, cached: Rollout) -> None:
    if cached.rollout_id != request.rollout_id:
        raise ValueError(
            f"cache collision for {request.cache_key}: expected rollout_id "
            f"{request.rollout_id}, found {cached.rollout_id}"
        )
    if (
        cached.prompt_id != request.prompt.prompt_id
        or cached.condition is not request.condition
        or cached.seed != request.seed
        or cached.sample_index != request.sample_index
    ):
        raise ValueError(f"cached rollout does not match request {request.rollout_id}")
    stored_key = cached.metadata.get("cache_key")
    if stored_key is not None and stored_key != request.cache_key:
        raise ValueError(f"cached rollout has a mismatched embedded cache key: {request.rollout_id}")


def generate_rollout_batch(
    runner: ModelRunner,
    prompts: Sequence[Prompt],
    *,
    samples_per_prompt: int | Mapping[PromptStrategy | str, int],
    parameters: GenerationParameters = DEFAULT_ROLLOUT_PARAMETERS,
    base_seed: int = 1001,
    generation_config_version: str = "rollout-v1",
    cache: MutableMapping[str, Rollout | Mapping[str, Any]] | None = None,
) -> RolloutBatch:
    """Generate a batch, consulting an optional caller-owned content cache."""

    composition = _composition(runner)
    requests = build_rollout_requests(
        runner,
        prompts,
        samples_per_prompt=samples_per_prompt,
        parameters=parameters,
        base_seed=base_seed,
        generation_config_version=generation_config_version,
    )
    rollouts: list[Rollout] = []
    generated_count = 0
    cache_hit_count = 0
    for request in requests:
        if cache is not None and request.cache_key in cache:
            cached = _coerce_cached(cache[request.cache_key])
            _validate_cached(request, cached)
            rollouts.append(cached)
            cache_hit_count += 1
            continue

        result = runner.generate(
            [message.to_dict() for message in request.prompt.messages],
            parameters=parameters,
            seed=request.seed,
        )
        rollout = Rollout(
            rollout_id=request.rollout_id,
            prompt_id=request.prompt.prompt_id,
            condition=request.condition,
            base_model=str(composition["base_model"]),
            adapter_name=composition["adapter_name"],
            adapter_active=bool(composition["adapter_active"]),
            meta_ia_name=composition["meta_ia_name"],
            meta_ia_active=bool(composition["meta_ia_active"]),
            seed=request.seed,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            max_new_tokens=parameters.max_new_tokens,
            sample_index=request.sample_index,
            generation_config_version=generation_config_version,
            response=result.response,
            metadata={
                "cache_key": request.cache_key,
                "input_tokens": result.input_tokens,
                "generated_tokens": result.generated_tokens,
            },
        )
        if cache is not None:
            cache[request.cache_key] = rollout
        rollouts.append(rollout)
        generated_count += 1
    return RolloutBatch(tuple(rollouts), generated_count, cache_hit_count)


def generate_rollouts(
    runner: ModelRunner,
    prompts: Sequence[Prompt],
    *,
    samples_per_prompt: int | Mapping[PromptStrategy | str, int],
    parameters: GenerationParameters = DEFAULT_ROLLOUT_PARAMETERS,
    base_seed: int = 1001,
    generation_config_version: str = "rollout-v1",
    cache: MutableMapping[str, Rollout | Mapping[str, Any]] | None = None,
) -> tuple[Rollout, ...]:
    """Convenience wrapper returning only records from :func:`generate_rollout_batch`."""

    return generate_rollout_batch(
        runner,
        prompts,
        samples_per_prompt=samples_per_prompt,
        parameters=parameters,
        base_seed=base_seed,
        generation_config_version=generation_config_version,
        cache=cache,
    ).rollouts


__all__ = [
    "DEFAULT_ROLLOUT_PARAMETERS",
    "RolloutBatch",
    "RolloutRequest",
    "build_rollout_requests",
    "generate_rollout_batch",
    "generate_rollouts",
    "make_rollout_cache_key",
    "make_rollout_id",
    "rollout_cache_key",
    "sample_seed",
]
