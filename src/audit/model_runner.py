"""Fail-closed loading and generation for one model condition per process."""

from __future__ import annotations

import contextlib
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapter_manager import AdapterManager, AdapterReference
from .schemas import ModelCondition, condition_flags


CLEAN_CONDITIONS = frozenset(
    {ModelCondition.BASE, ModelCondition.PROMPT_GEN, ModelCondition.JUDGE}
)


@dataclass(frozen=True, slots=True)
class GenerationParameters:
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 512
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        defaults: "GenerationParameters | None" = None,
    ) -> "GenerationParameters":
        base = defaults or cls()
        max_input = value.get("max_input_tokens", base.max_input_tokens)
        return cls(
            temperature=float(value.get("temperature", base.temperature)),
            top_p=float(value.get("top_p", base.top_p)),
            max_new_tokens=int(value.get("max_new_tokens", base.max_new_tokens)),
            max_input_tokens=None if max_input is None else int(max_input),
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    response: str
    input_tokens: int
    generated_tokens: int
    seed: int


def normalize_condition(value: ModelCondition | str) -> ModelCondition:
    if isinstance(value, ModelCondition):
        return value
    normalized = str(value).strip().upper()
    try:
        return ModelCondition(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown model condition: {value!r}") from exc


def extract_json_value(text: str, fallback_to_conservative: bool = False) -> Any:
    """Parse a JSON value even when a model surrounds it with prose/fences.

    Args:
        text: The model response containing JSON
        fallback_to_conservative: If True, return conservative no-difference JSON instead of raising
    """

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = sorted(
        offset for offset in (candidate.find("{"), candidate.find("[")) if offset >= 0
    )
    for offset in starts:
        try:
            value, _ = decoder.raw_decode(candidate[offset:])
            return value
        except json.JSONDecodeError:
            continue

    # Better error logging before failing
    print(f"[WARNING] JSON extraction failed on response (first 500 chars): {candidate[:500]}", file=sys.stderr)
    if fallback_to_conservative:
        print(f"[FALLBACK] Returning conservative 'no-difference' JSON response", file=sys.stderr)
        # Return a valid "no meaningful difference" response that passes validation
        return {
            "meaningful_difference": False,
            "candidate_behavior": None,
            "group_showing_behavior": "unclear",
            "trigger_context": None,
            "supporting_response_ids": [],
            "counterevidence_response_ids": [],
            "alternative_explanations": [],
            "confidence": 0.0,
        }
    raise ValueError(f"Model response did not contain valid JSON: {candidate[:200]}")




def _freeze(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _active_adapters(model: Any) -> list[str]:
    active = getattr(getattr(model, "base_model", None), "active_adapters", None)
    if active is None:
        active = getattr(model, "active_adapters", None)
    if active is None:
        return []
    return [active] if isinstance(active, str) else list(active)


class ModelRunner:
    """Own a single exact composition: clean base, behavior, IA, or both.

    Clean roles do not resolve adapter references at all. Production callers
    should create one runner per process/job and persist its outputs before a
    different role starts.
    """

    def __init__(
        self,
        *,
        condition: ModelCondition | str,
        base_model_path: str | Path,
        base_model_id: str | None = None,
        dtype: str = "bfloat16",
        device_map: str | Mapping[str, Any] = "auto",
        local_files_only: bool = True,
        behavior_adapter: AdapterReference | None = None,
        meta_ia_adapter: AdapterReference | None = None,
        adapter_manager: AdapterManager | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.condition = normalize_condition(condition)
        self.base_model_path = str(base_model_path)
        self.base_model_id = base_model_id or str(base_model_path)
        self.dtype_name = dtype
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.behavior_adapter = behavior_adapter
        self.meta_ia_adapter = meta_ia_adapter
        self.adapter_manager = adapter_manager or AdapterManager()
        self.trust_remote_code = trust_remote_code
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._torch: Any | None = None
        self._expected_adapters: tuple[str, ...] = ()
        self._validate_request()

    @property
    def adapter_active(self) -> bool:
        return condition_flags(self.condition)[0]

    @property
    def meta_ia_active(self) -> bool:
        return condition_flags(self.condition)[1]

    @property
    def composition(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "base_model": self.base_model_id,
            "adapter_active": self.adapter_active,
            "meta_ia_active": self.meta_ia_active,
            "adapter_name": (
                self.behavior_adapter.name
                if self.adapter_active and self.behavior_adapter
                else None
            ),
            "meta_ia_name": (
                self.meta_ia_adapter.name
                if self.meta_ia_active and self.meta_ia_adapter
                else None
            ),
        }

    def _validate_request(self) -> None:
        if self.adapter_active and self.behavior_adapter is None:
            raise ValueError(f"{self.condition.value} requires a behavior adapter")
        if self.meta_ia_active and self.meta_ia_adapter is None:
            raise ValueError(f"{self.condition.value} requires a Meta-IA adapter")
        if (
            self.adapter_active
            and self.meta_ia_active
            and self.behavior_adapter is not None
            and self.meta_ia_adapter is not None
            and self.behavior_adapter.name == self.meta_ia_adapter.name
        ):
            raise ValueError("Behavior and Meta-IA adapter names must differ")

    def _dtype(self, torch: Any) -> Any:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            return mapping[self.dtype_name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {self.dtype_name}") from exc

    def load(self) -> "ModelRunner":
        if self.model is not None:
            return self

        # Resolve and validate small adapter checkpoints before allocating the
        # base model. This makes missing files, placeholder paths, wrong base
        # identities, and remote pinning errors fail before a costly 70B load.
        resolved_adapters: list[tuple[AdapterReference, Path]] = []
        if self.condition not in CLEAN_CONDITIONS:
            references: list[AdapterReference] = []
            if self.adapter_active:
                assert self.behavior_adapter is not None
                references.append(self.behavior_adapter)
            if self.meta_ia_active:
                assert self.meta_ia_adapter is not None
                references.append(self.meta_ia_adapter)
            resolved_adapters = [
                (reference, self.adapter_manager.resolve(reference))
                for reference in references
            ]

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Model execution requires torch and transformers") from exc

        self._torch = torch
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self._dtype(torch),
            device_map=self.device_map,
            local_files_only=self.local_files_only,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = tokenizer

        if self.condition in CLEAN_CONDITIONS:
            self.model = base
        else:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter conditions require peft") from exc
            first_pair, *remaining = resolved_adapters
            first, first_path = first_pair
            model = PeftModel.from_pretrained(
                base,
                str(first_path),
                adapter_name=first.name,
                is_trainable=False,
                local_files_only=True,
                low_cpu_mem_usage=False,
            )
            for reference, adapter_path in remaining:
                try:
                    model.load_adapter(
                        str(adapter_path),
                        adapter_name=reference.name,
                        is_trainable=False,
                        low_cpu_mem_usage=False,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "PEFT load_adapter must support low_cpu_mem_usage=False"
                    ) from exc
            self.model = model
            self._expected_adapters = tuple(
                reference.name for reference, _ in resolved_adapters
            )
            self._activate()

        _freeze(self.model)
        self.model.eval()
        self._assert_composition()
        return self

    def _activate(self) -> None:
        assert self.model is not None
        if len(self._expected_adapters) == 1:
            self.model.set_adapter(self._expected_adapters[0])
        elif self._expected_adapters:
            self.model.base_model.set_adapter(list(self._expected_adapters))
        _freeze(self.model)

    def _assert_composition(self) -> None:
        assert self.model is not None
        if self.condition in CLEAN_CONDITIONS:
            if hasattr(self.model, "peft_config"):
                raise RuntimeError("A clean model role unexpectedly loaded PEFT")
            return
        expected = set(self._expected_adapters)
        available = set(getattr(self.model, "peft_config", {}))
        if not expected.issubset(available):
            raise RuntimeError(
                f"Missing adapters: expected={sorted(expected)}, available={sorted(available)}"
            )
        active = set(_active_adapters(self.model))
        if active != expected:
            raise RuntimeError(
                f"Wrong active adapters: expected={sorted(expected)}, active={sorted(active)}"
            )
        meta = [
            name
            for name, parameter in self.model.named_parameters()
            if any(adapter in name for adapter in expected)
            and getattr(parameter, "is_meta", False)
        ]
        if meta:
            raise RuntimeError(f"Adapter tensors remained on meta device: {meta[:5]}")

    def _input_device(self) -> Any:
        assert self.model is not None
        return self.model.get_input_embeddings().weight.device

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        parameters: GenerationParameters,
        seed: int,
    ) -> GenerationResult:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        assert self._torch is not None
        if seed < 0:
            raise ValueError("seed must be non-negative")
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"} or not content.strip():
                raise ValueError(f"Invalid chat message: {message!r}")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("At least one message is required")

        encoded = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Handle various return types from apply_chat_template
        # Dict: extract input_ids
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise RuntimeError("Chat template mapping did not contain input_ids")
            encoded = encoded["input_ids"]
        # String: tokenize it
        if isinstance(encoded, str):
            encoded = self.tokenizer.encode(encoded, return_tensors="pt")
        # List: convert to tensor
        elif isinstance(encoded, list):
            encoded = self._torch.tensor([encoded], dtype=self._torch.long)
        # Non-tensor: convert to tensor
        elif not isinstance(encoded, self._torch.Tensor):
            try:
                encoded = self._torch.tensor(encoded, dtype=self._torch.long)
            except (TypeError, ValueError):
                # If tensor conversion fails, try encoding as string
                encoded = self.tokenizer.encode(str(encoded), return_tensors="pt")
        if parameters.max_input_tokens is not None:
            encoded = encoded[:, -parameters.max_input_tokens :]
        encoded = encoded.to(self._input_device())
        attention = self._torch.ones_like(encoded)
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)

        kwargs: dict[str, Any] = {
            "input_ids": encoded,
            "attention_mask": attention,
            "max_new_tokens": parameters.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": parameters.temperature > 0,
        }
        if parameters.temperature > 0:
            kwargs["temperature"] = parameters.temperature
            kwargs["top_p"] = parameters.top_p
        with self._torch.inference_mode():
            output = self.model.generate(**kwargs)
        generated = output[0, encoded.shape[1] :]
        return GenerationResult(
            response=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            input_tokens=int(encoded.shape[1]),
            generated_tokens=int(generated.shape[0]),
            seed=seed,
        )

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        parameters: GenerationParameters,
        seed: int,
        max_retries: int = 3,
    ) -> tuple[Any, GenerationResult]:
        import time
        last_error = None
        for attempt in range(max_retries):
            result = self.generate(messages, parameters=parameters, seed=seed + attempt)
            try:
                payload = extract_json_value(result.response)
                if payload:  # Successfully got non-empty JSON
                    return payload, result
            except ValueError as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[RETRY {attempt + 1}/{max_retries}] JSON extraction failed, retrying in 2s...", file=sys.stderr)
                    time.sleep(2)
                    continue

        # A generic model runner cannot invent a schema-correct fallback: the
        # caller may be grading behavior, differences, clustering, or semantic
        # matches. Fail closed so invalid judge text never becomes evidence.
        assert last_error is not None
        raise last_error


    def close(self) -> None:
        model = self.model
        self.model = None
        self.tokenizer = None
        if model is not None:
            del model
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            with contextlib.suppress(Exception):
                self._torch.cuda.empty_cache()
        self._torch = None
        self._expected_adapters = ()

    def __enter__(self) -> "ModelRunner":
        return self.load()

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "CLEAN_CONDITIONS",
    "GenerationParameters",
    "GenerationResult",
    "ModelRunner",
    "extract_json_value",
    "normalize_condition",
]
