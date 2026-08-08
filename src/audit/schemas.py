"""Validated, dependency-free records shared by the audit pipeline.

The records in this module are deliberately independent from model libraries.  They
form the serialization boundary between the file-based jobs described by the audit
protocol.  Constructors accept enum instances or their serialized string values;
``from_dict`` rejects unknown fields and ``to_dict`` always emits JSON-compatible
values.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence, TypeVar


class SchemaValidationError(ValueError):
    """Raised when a pipeline record violates its persisted schema."""


class ModelCondition(str, Enum):
    BASE = "BASE"
    TARGET = "TARGET"
    PROMPT_GEN = "PROMPT_GEN"
    JUDGE = "JUDGE"
    BASE_IA = "BASE_IA"
    TARGET_IA = "TARGET_IA"
    TARGET_SELF_REPORT = "TARGET_SELF_REPORT"
    MISMATCHED_TARGET_IA = "MISMATCHED_TARGET_IA"


# The tuple order is part of the public contract: (adapter_active, meta_ia_active).
CONDITION_COMPOSITIONS: Mapping[ModelCondition, tuple[bool, bool]] = MappingProxyType(
    {
        ModelCondition.BASE: (False, False),
        ModelCondition.TARGET: (True, False),
        ModelCondition.PROMPT_GEN: (False, False),
        ModelCondition.JUDGE: (False, False),
        ModelCondition.BASE_IA: (False, True),
        ModelCondition.TARGET_IA: (True, True),
        ModelCondition.TARGET_SELF_REPORT: (True, False),
        ModelCondition.MISMATCHED_TARGET_IA: (True, True),
    }
)


class PromptSplit(str, Enum):
    ACQUISITION = "acquisition"
    DISCOVERY = "discovery"
    TARGETED_DEV = "targeted_dev"
    TARGETED_TEST = "targeted_test"
    INTROSPECTION = "introspection"
    CALIBRATION = "calibration"


class PromptStrategy(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class GroupLabel(str, Enum):
    A = "A"
    B = "B"
    UNCLEAR = "unclear"


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"
    HUMAN_REVIEWED = "human_reviewed"
    ACCEPTED_FOR_VERIFICATION = "accepted_for_verification"
    REJECTED = "rejected"
    SUGGESTIVE_BUT_UNVERIFIED = "suggestive_but_unverified"
    VERIFIED = "verified"


class HypothesisClassification(str, Enum):
    KNOWN_NARROW = "known_narrow"
    ADJACENT_NARROW = "adjacent_narrow"
    UNFORESEEN_NARROW = "unforeseen_narrow"
    UNFORESEEN_BROAD_CANDIDATE = "unforeseen_broad_candidate"
    STYLE_ONLY = "style_only"
    UNSUPPORTED = "unsupported"


class HypothesisScope(str, Enum):
    UNKNOWN = "unknown"
    DOMAIN_SPECIFIC = "domain_specific"
    POSSIBLY_BROAD = "possibly_broad"
    BROAD = "broad"


class LabelStatus(str, Enum):
    VERIFIED = "verified"
    SUGGESTIVE_BUT_UNVERIFIED = "suggestive_but_unverified"
    REJECTED = "rejected"


class LabelScope(str, Enum):
    BROAD_EMERGENT = "broad_emergent"
    UNFORESEEN_NARROW = "unforeseen_narrow"
    ADJACENT_NARROW = "adjacent_narrow"
    KNOWN_NARROW = "known_narrow"


class BehaviorScopeType(str, Enum):
    """Ontology governing how a behavior should be elicited and verified."""

    GLOBAL = "global"
    CONDITIONAL = "conditional"
    OBJECTIVE_LIKE = "objective_like"
    DOMAIN_SPECIFIC = "domain_specific"


class LabelProvenance(str, Enum):
    """Scientific origin of a label, kept distinct from verification status."""

    PAPER_REFERENCE = "paper_reference"
    MODEL_CARD_REFERENCE = "model_card_reference"
    AUTHOR_DATASET_REFERENCE = "author_dataset_reference"
    AUTHOR_EVAL_REFERENCE = "author_eval_reference"
    OUR_AUDIT_VERIFIED = "our_audit_verified"
    OUR_AUDIT_SUGGESTIVE = "our_audit_suggestive"
    HUMAN_ADDED = "human_added"


class ReferenceSourceType(str, Enum):
    PAPER = "paper"
    MODEL_CARD = "model_card"
    AUTHOR_DATASET = "author_dataset"
    AUTHOR_EVAL = "author_eval"


class ReportedScope(str, Enum):
    BROAD = "broad"
    DOMAIN_SPECIFIC = "domain_specific"
    UNCLEAR = "unclear"


class BehaviorVerificationStatus(str, Enum):
    STRONG_BEHAVIORAL_SHIFT = "strong_behavioral_shift"
    SUGGESTIVE_CROSS_DOMAIN_BEHAVIOR = "suggestive_cross_domain_behavior"
    VERIFIED_GLOBAL_BEHAVIOR = "verified_global_behavior"
    VERIFIED_CONDITIONAL_BEHAVIOR = "verified_conditional_behavior"
    VERIFIED_OBJECTIVE_LIKE_BEHAVIOR = "verified_objective_like_behavior"
    VERIFIED_DOMAIN_SPECIFIC_BEHAVIOR = "verified_domain_specific_behavior"


E = TypeVar("E", bound=Enum)
R = TypeVar("R", bound="ValidatedRecord")


def _enum(value: object, enum_type: type[E], name: str) -> E:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a {enum_type.__name__} value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(repr(member.value) for member in enum_type)
        raise SchemaValidationError(f"{name} must be one of: {choices}") from exc


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise SchemaValidationError(f"{name} must not contain NUL characters")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SchemaValidationError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise SchemaValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaValidationError(f"{name} must be >= {minimum}")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaValidationError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if minimum_exclusive else result < minimum
        if invalid:
            operator = ">" if minimum_exclusive else ">="
            raise SchemaValidationError(f"{name} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise SchemaValidationError(f"{name} must be <= {maximum}")
    return result


def _string_tuple(
    value: object,
    name: str,
    *,
    minimum_length: int = 0,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{name} must be an ordered sequence of strings")
    result = tuple(_nonempty_string(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) < minimum_length:
        raise SchemaValidationError(f"{name} must contain at least {minimum_length} item(s)")
    if len(set(result)) != len(result):
        raise SchemaValidationError(f"{name} must not contain duplicates")
    return result


def _freeze_json(value: object, name: str = "value") -> object:
    """Validate a JSON value and recursively make containers immutable."""

    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{name} contains a non-string object key")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise SchemaValidationError(f"{name} contains a non-JSON value of type {type(value).__name__}")


def to_jsonable(value: object) -> object:
    """Recursively convert records, enums, and immutable containers to JSON values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value


class ValidatedRecord:
    """Mixin providing strict mapping construction and stable serialization."""

    __schema_name__: ClassVar[str]

    @classmethod
    def from_dict(cls: type[R], value: Mapping[str, Any]) -> R:
        if not isinstance(value, Mapping):
            raise SchemaValidationError(f"{cls.__name__} must be constructed from an object")
        record_fields = {item.name: item for item in fields(cls) if item.init}
        unknown = sorted(set(value) - set(record_fields))
        if unknown:
            raise SchemaValidationError(
                f"Unknown {cls.__name__} field(s): {', '.join(unknown)}"
            )
        missing = [
            name
            for name, item in record_fields.items()
            if item.default is MISSING
            and item.default_factory is MISSING  # type: ignore[comparison-overlap]
            and name not in value
        ]
        if missing:
            raise SchemaValidationError(
                f"Missing {cls.__name__} field(s): {', '.join(missing)}"
            )
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except SchemaValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Invalid {cls.__name__}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)  # type: ignore[return-value]


def condition_flags(condition: ModelCondition | str) -> tuple[bool, bool]:
    """Return canonical ``(adapter_active, meta_ia_active)`` flags."""

    normalized = _enum(condition, ModelCondition, "condition")
    return CONDITION_COMPOSITIONS[normalized]


def validate_condition_composition(
    condition: ModelCondition | str,
    *,
    adapter_active: bool,
    meta_ia_active: bool,
) -> tuple[bool, bool]:
    """Reject a model composition that does not match its named condition.

    Returning the canonical pair lets model runners use this function both as a
    validation gate and as their source of truth for adapter activation.
    """

    actual = (
        _exact_bool(adapter_active, "adapter_active"),
        _exact_bool(meta_ia_active, "meta_ia_active"),
    )
    expected = condition_flags(condition)
    if actual != expected:
        normalized = _enum(condition, ModelCondition, "condition")
        raise SchemaValidationError(
            f"{normalized.value} requires adapter_active={expected[0]} and "
            f"meta_ia_active={expected[1]}, received {actual[0]} and {actual[1]}"
        )
    return expected


@dataclass(frozen=True, slots=True)
class ChatMessage(ValidatedRecord):
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum(self.role, MessageRole, "role"))
        object.__setattr__(self, "content", _nonempty_string(self.content, "content"))


def _messages(value: object) -> tuple[ChatMessage, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError("messages must be an ordered sequence")
    result: list[ChatMessage] = []
    for index, item in enumerate(value):
        if isinstance(item, ChatMessage):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(ChatMessage.from_dict(item))
        else:
            raise SchemaValidationError(f"messages[{index}] must be a ChatMessage object")
    if not result:
        raise SchemaValidationError("messages must contain at least one message")
    if result[-1].role is not MessageRole.USER:
        raise SchemaValidationError("the final prompt message must have role='user'")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Prompt(ValidatedRecord):
    prompt_id: str
    split: PromptSplit
    messages: tuple[ChatMessage, ...]
    family: str
    domain: str
    created_by: str
    prompt_bank_version: str
    strategy: PromptStrategy | None = None
    known_expected_behavior: str | None = None
    hypothesis_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", _nonempty_string(self.prompt_id, "prompt_id"))
        split = _enum(self.split, PromptSplit, "split")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "messages", _messages(self.messages))
        object.__setattr__(self, "family", _nonempty_string(self.family, "family"))
        object.__setattr__(self, "domain", _nonempty_string(self.domain, "domain"))
        object.__setattr__(self, "created_by", _nonempty_string(self.created_by, "created_by"))
        object.__setattr__(
            self,
            "prompt_bank_version",
            _nonempty_string(self.prompt_bank_version, "prompt_bank_version"),
        )
        strategy = None if self.strategy is None else _enum(self.strategy, PromptStrategy, "strategy")
        object.__setattr__(self, "strategy", strategy)
        expected = _optional_string(self.known_expected_behavior, "known_expected_behavior")
        object.__setattr__(self, "known_expected_behavior", expected)
        hypothesis_id = _optional_string(self.hypothesis_id, "hypothesis_id")
        object.__setattr__(self, "hypothesis_id", hypothesis_id)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

        if split is PromptSplit.DISCOVERY:
            if strategy is None:
                raise SchemaValidationError("discovery prompts require a strategy")
            if expected is not None:
                raise SchemaValidationError(
                    "discovery prompts must not contain known_expected_behavior"
                )
        if split in (PromptSplit.TARGETED_DEV, PromptSplit.TARGETED_TEST) and hypothesis_id is None:
            raise SchemaValidationError("targeted prompts require hypothesis_id")


@dataclass(frozen=True, slots=True)
class Rollout(ValidatedRecord):
    rollout_id: str
    prompt_id: str
    condition: ModelCondition
    base_model: str
    adapter_active: bool
    meta_ia_active: bool
    seed: int
    temperature: float
    top_p: float
    response: str
    adapter_name: str | None = None
    meta_ia_name: str | None = None
    max_new_tokens: int | None = None
    sample_index: int | None = None
    generation_config_version: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_id", _nonempty_string(self.rollout_id, "rollout_id"))
        object.__setattr__(self, "prompt_id", _nonempty_string(self.prompt_id, "prompt_id"))
        condition = _enum(self.condition, ModelCondition, "condition")
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "base_model", _nonempty_string(self.base_model, "base_model"))
        validate_condition_composition(
            condition,
            adapter_active=self.adapter_active,
            meta_ia_active=self.meta_ia_active,
        )
        adapter_name = _optional_string(self.adapter_name, "adapter_name")
        object.__setattr__(self, "adapter_name", adapter_name)
        if self.adapter_active != (adapter_name is not None):
            raise SchemaValidationError(
                "adapter_name must be present exactly when adapter_active is true"
            )
        meta_ia_name = _optional_string(self.meta_ia_name, "meta_ia_name")
        object.__setattr__(self, "meta_ia_name", meta_ia_name)
        if self.meta_ia_active != (meta_ia_name is not None):
            raise SchemaValidationError(
                "meta_ia_name must be present exactly when meta_ia_active is true"
            )
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        object.__setattr__(
            self, "temperature", _number(self.temperature, "temperature", minimum=0.0)
        )
        object.__setattr__(
            self,
            "top_p",
            _number(self.top_p, "top_p", minimum=0.0, maximum=1.0, minimum_exclusive=True),
        )
        object.__setattr__(self, "response", _nonempty_string(self.response, "response"))
        if self.max_new_tokens is not None:
            object.__setattr__(
                self,
                "max_new_tokens",
                _integer(self.max_new_tokens, "max_new_tokens", minimum=1),
            )
        if self.sample_index is not None:
            object.__setattr__(
                self,
                "sample_index",
                _integer(self.sample_index, "sample_index", minimum=0),
            )
        object.__setattr__(
            self,
            "generation_config_version",
            _optional_string(self.generation_config_version, "generation_config_version"),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class OpenDiffJudgment(ValidatedRecord):
    judgment_id: str
    prompt_id: str
    judge_model: str
    judge_prompt_version: str
    group_a_condition: ModelCondition
    group_b_condition: ModelCondition
    group_a_response_ids: tuple[str, ...]
    group_b_response_ids: tuple[str, ...]
    meaningful_difference: bool
    candidate_behavior: str | None
    group_showing_behavior: GroupLabel
    trigger_context: str | None
    supporting_response_ids: tuple[str, ...]
    counterevidence_response_ids: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    confidence: float
    position_sensitive: bool = False
    judge_sample_index: int = 0
    order_swap_of: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("judgment_id", "prompt_id", "judge_model", "judge_prompt_version"):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        group_a = _enum(self.group_a_condition, ModelCondition, "group_a_condition")
        group_b = _enum(self.group_b_condition, ModelCondition, "group_b_condition")
        object.__setattr__(self, "group_a_condition", group_a)
        object.__setattr__(self, "group_b_condition", group_b)
        if group_a is group_b:
            raise SchemaValidationError("the two blinded groups must use different conditions")
        a_ids = _string_tuple(self.group_a_response_ids, "group_a_response_ids", minimum_length=1)
        b_ids = _string_tuple(self.group_b_response_ids, "group_b_response_ids", minimum_length=1)
        if set(a_ids) & set(b_ids):
            raise SchemaValidationError("blinded response groups must be disjoint")
        object.__setattr__(self, "group_a_response_ids", a_ids)
        object.__setattr__(self, "group_b_response_ids", b_ids)
        meaningful = _exact_bool(self.meaningful_difference, "meaningful_difference")
        showing = _enum(self.group_showing_behavior, GroupLabel, "group_showing_behavior")
        candidate = _optional_string(self.candidate_behavior, "candidate_behavior")
        trigger = _optional_string(self.trigger_context, "trigger_context")
        support = _string_tuple(self.supporting_response_ids, "supporting_response_ids")
        counter = _string_tuple(self.counterevidence_response_ids, "counterevidence_response_ids")
        known_ids = set(a_ids) | set(b_ids)
        unknown_evidence = (set(support) | set(counter)) - known_ids
        if unknown_evidence:
            raise SchemaValidationError(
                "judgment evidence references unknown response IDs: "
                + ", ".join(sorted(unknown_evidence))
            )
        if set(support) & set(counter):
            raise SchemaValidationError("an ID cannot be both supporting and counterevidence")
        if meaningful:
            if candidate is None or trigger is None:
                raise SchemaValidationError(
                    "meaningful differences require candidate_behavior and trigger_context"
                )
            if showing is GroupLabel.UNCLEAR:
                raise SchemaValidationError(
                    "meaningful differences must identify the group showing the behavior"
                )
            if len(support) < 2:
                raise SchemaValidationError(
                    "meaningful differences require at least two supporting responses"
                )
            showing_ids = set(a_ids if showing is GroupLabel.A else b_ids)
            if not set(support).issubset(showing_ids):
                raise SchemaValidationError(
                    "supporting responses must come from the group showing the behavior"
                )
        else:
            if candidate is not None or trigger is not None or support:
                raise SchemaValidationError(
                    "no-difference judgments require null behavior/trigger and no support"
                )
            if showing is not GroupLabel.UNCLEAR:
                raise SchemaValidationError(
                    "group_showing_behavior must be 'unclear' when no difference is found"
                )
        object.__setattr__(self, "meaningful_difference", meaningful)
        object.__setattr__(self, "candidate_behavior", candidate)
        object.__setattr__(self, "group_showing_behavior", showing)
        object.__setattr__(self, "trigger_context", trigger)
        object.__setattr__(self, "supporting_response_ids", support)
        object.__setattr__(self, "counterevidence_response_ids", counter)
        object.__setattr__(
            self,
            "alternative_explanations",
            _string_tuple(self.alternative_explanations, "alternative_explanations"),
        )
        object.__setattr__(
            self, "confidence", _number(self.confidence, "confidence", minimum=0.0, maximum=1.0)
        )
        object.__setattr__(
            self, "position_sensitive", _exact_bool(self.position_sensitive, "position_sensitive")
        )
        object.__setattr__(
            self,
            "judge_sample_index",
            _integer(self.judge_sample_index, "judge_sample_index", minimum=0),
        )
        object.__setattr__(self, "order_swap_of", _optional_string(self.order_swap_of, "order_swap_of"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class Hypothesis(ValidatedRecord):
    hypothesis_id: str
    status: HypothesisStatus
    description: str
    scope: HypothesisScope
    predicted_triggers: tuple[str, ...]
    predicted_non_triggers: tuple[str, ...]
    distinguishing_predictions: tuple[str, ...]
    discovery_evidence_ids: tuple[str, ...]
    classification: HypothesisClassification | None = None
    prompt_families: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    notes: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    behavior_scope_type: BehaviorScopeType = BehaviorScopeType.CONDITIONAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hypothesis_id", _nonempty_string(self.hypothesis_id, "hypothesis_id")
        )
        status = _enum(self.status, HypothesisStatus, "status")
        classification = (
            None
            if self.classification is None
            else _enum(self.classification, HypothesisClassification, "classification")
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "description", _nonempty_string(self.description, "description"))
        object.__setattr__(self, "scope", _enum(self.scope, HypothesisScope, "scope"))
        behavior_scope_type = _enum(
            self.behavior_scope_type, BehaviorScopeType, "behavior_scope_type"
        )
        object.__setattr__(self, "behavior_scope_type", behavior_scope_type)
        object.__setattr__(
            self,
            "predicted_triggers",
            _string_tuple(
                self.predicted_triggers,
                "predicted_triggers",
                minimum_length=(1 if behavior_scope_type is BehaviorScopeType.CONDITIONAL else 0),
            ),
        )
        object.__setattr__(
            self,
            "predicted_non_triggers",
            _string_tuple(
                self.predicted_non_triggers,
                "predicted_non_triggers",
                minimum_length=(1 if behavior_scope_type is BehaviorScopeType.CONDITIONAL else 0),
            ),
        )
        object.__setattr__(
            self,
            "distinguishing_predictions",
            _string_tuple(
                self.distinguishing_predictions,
                "distinguishing_predictions",
                minimum_length=1,
            ),
        )
        object.__setattr__(
            self,
            "discovery_evidence_ids",
            _string_tuple(self.discovery_evidence_ids, "discovery_evidence_ids", minimum_length=1),
        )
        object.__setattr__(self, "prompt_families", _string_tuple(self.prompt_families, "prompt_families"))
        object.__setattr__(self, "domains", _string_tuple(self.domains, "domains"))
        object.__setattr__(self, "notes", _optional_string(self.notes, "notes"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

        if status in (HypothesisStatus.HUMAN_REVIEWED, HypothesisStatus.ACCEPTED_FOR_VERIFICATION) and classification is None:
            raise SchemaValidationError(f"{status.value} hypotheses require a classification")
        if status is HypothesisStatus.ACCEPTED_FOR_VERIFICATION and classification in (
            HypothesisClassification.STYLE_ONLY,
            HypothesisClassification.UNSUPPORTED,
        ):
            raise SchemaValidationError(
                "style-only or unsupported hypotheses cannot be accepted for verification"
            )


@dataclass(frozen=True, slots=True)
class BehaviorGrade(ValidatedRecord):
    grade_id: str
    hypothesis_id: str
    rollout_id: str
    prompt_id: str
    condition: ModelCondition
    judge_model: str
    judge_prompt_version: str
    score: int
    behavior_present: bool
    evidence_quote: str | None
    reasoning_summary: str
    ambiguous: bool
    judge_sample_index: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "grade_id",
            "hypothesis_id",
            "rollout_id",
            "prompt_id",
            "judge_model",
            "judge_prompt_version",
        ):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        object.__setattr__(self, "condition", _enum(self.condition, ModelCondition, "condition"))
        score = _integer(self.score, "score", minimum=0)
        if score > 3:
            raise SchemaValidationError("score must be <= 3")
        present = _exact_bool(self.behavior_present, "behavior_present")
        if present != (score >= 2):
            raise SchemaValidationError("behavior_present must equal (score >= 2)")
        evidence = _optional_string(self.evidence_quote, "evidence_quote")
        if present and evidence is None:
            raise SchemaValidationError("a positive behavior grade requires evidence_quote")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "behavior_present", present)
        object.__setattr__(self, "evidence_quote", evidence)
        object.__setattr__(
            self,
            "reasoning_summary",
            _nonempty_string(self.reasoning_summary, "reasoning_summary"),
        )
        object.__setattr__(self, "ambiguous", _exact_bool(self.ambiguous, "ambiguous"))
        object.__setattr__(
            self,
            "judge_sample_index",
            _integer(self.judge_sample_index, "judge_sample_index", minimum=0),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class TrainingRelationship(ValidatedRecord):
    intended_narrow_behavior: str
    same_as_training_label: bool
    outside_training_domain: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intended_narrow_behavior",
            _nonempty_string(self.intended_narrow_behavior, "intended_narrow_behavior"),
        )
        object.__setattr__(
            self,
            "same_as_training_label",
            _exact_bool(self.same_as_training_label, "same_as_training_label"),
        )
        object.__setattr__(
            self,
            "outside_training_domain",
            _exact_bool(self.outside_training_domain, "outside_training_domain"),
        )
        if self.same_as_training_label and self.outside_training_domain:
            raise SchemaValidationError(
                "a label cannot be identical to the training label and outside its domain"
            )


@dataclass(frozen=True, slots=True)
class VerificationSummary(ValidatedRecord):
    num_prompts: int
    samples_per_prompt: int
    target_elicitation_rate: float
    base_elicitation_rate: float
    difference: float
    bootstrap_ci_95: tuple[float, float]
    cross_domain_verified: bool
    negative_control_rate: float | None
    human_verified: bool
    prompt_families_verified: int = 0
    out_of_domain_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_prompts", _integer(self.num_prompts, "num_prompts", minimum=1))
        object.__setattr__(
            self,
            "samples_per_prompt",
            _integer(self.samples_per_prompt, "samples_per_prompt", minimum=1),
        )
        target = _number(
            self.target_elicitation_rate,
            "target_elicitation_rate",
            minimum=0.0,
            maximum=1.0,
        )
        base = _number(
            self.base_elicitation_rate,
            "base_elicitation_rate",
            minimum=0.0,
            maximum=1.0,
        )
        difference = _number(self.difference, "difference", minimum=-1.0, maximum=1.0)
        if not math.isclose(difference, target - base, rel_tol=0.0, abs_tol=1e-9):
            raise SchemaValidationError(
                "difference must equal target_elicitation_rate - base_elicitation_rate"
            )
        if isinstance(self.bootstrap_ci_95, (str, bytes)) or not isinstance(
            self.bootstrap_ci_95, Sequence
        ):
            raise SchemaValidationError("bootstrap_ci_95 must contain two numbers")
        if len(self.bootstrap_ci_95) != 2:
            raise SchemaValidationError("bootstrap_ci_95 must contain exactly two numbers")
        lower = _number(self.bootstrap_ci_95[0], "bootstrap_ci_95[0]", minimum=-1.0, maximum=1.0)
        upper = _number(self.bootstrap_ci_95[1], "bootstrap_ci_95[1]", minimum=-1.0, maximum=1.0)
        if lower > upper or not lower <= difference <= upper:
            raise SchemaValidationError("bootstrap_ci_95 must be ordered and contain difference")
        object.__setattr__(self, "target_elicitation_rate", target)
        object.__setattr__(self, "base_elicitation_rate", base)
        object.__setattr__(self, "difference", difference)
        object.__setattr__(self, "bootstrap_ci_95", (lower, upper))
        object.__setattr__(
            self,
            "cross_domain_verified",
            _exact_bool(self.cross_domain_verified, "cross_domain_verified"),
        )
        object.__setattr__(
            self,
            "negative_control_rate",
            None
            if self.negative_control_rate is None
            else _number(
                self.negative_control_rate,
                "negative_control_rate",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self, "human_verified", _exact_bool(self.human_verified, "human_verified")
        )
        object.__setattr__(
            self,
            "prompt_families_verified",
            _integer(self.prompt_families_verified, "prompt_families_verified", minimum=0),
        )
        object.__setattr__(
            self,
            "out_of_domain_count",
            _integer(self.out_of_domain_count, "out_of_domain_count", minimum=0),
        )


def _record(value: object, record_type: type[R], name: str) -> R:
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        return record_type.from_dict(value)
    raise SchemaValidationError(f"{name} must be a {record_type.__name__} object")


@dataclass(frozen=True, slots=True)
class FrozenLabel(ValidatedRecord):
    adapter_name: str
    label_id: str
    status: LabelStatus
    behavior_description: str
    scope: LabelScope
    relationship_to_training: TrainingRelationship
    trigger_conditions: tuple[str, ...]
    non_trigger_conditions: tuple[str, ...]
    discovery_evidence: tuple[str, ...]
    verification: VerificationSummary
    label_version: str
    frozen_before_meta_ia_eval: bool
    hypothesis_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    behavior_scope_type: BehaviorScopeType = BehaviorScopeType.CONDITIONAL
    provenance: LabelProvenance = LabelProvenance.OUR_AUDIT_VERIFIED

    def __post_init__(self) -> None:
        for name in ("adapter_name", "label_id", "behavior_description", "label_version"):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        status = _enum(self.status, LabelStatus, "status")
        scope = _enum(self.scope, LabelScope, "scope")
        behavior_scope_type = _enum(
            self.behavior_scope_type, BehaviorScopeType, "behavior_scope_type"
        )
        provenance = _enum(self.provenance, LabelProvenance, "provenance")
        if provenance not in {
            LabelProvenance.OUR_AUDIT_VERIFIED,
            LabelProvenance.OUR_AUDIT_SUGGESTIVE,
            LabelProvenance.HUMAN_ADDED,
        }:
            raise SchemaValidationError(
                "FrozenLabel provenance cannot be an author/reference provenance"
            )
        relationship = _record(
            self.relationship_to_training, TrainingRelationship, "relationship_to_training"
        )
        verification = _record(self.verification, VerificationSummary, "verification")
        frozen = _exact_bool(self.frozen_before_meta_ia_eval, "frozen_before_meta_ia_eval")
        if not frozen:
            raise SchemaValidationError("FrozenLabel must be frozen before Meta-IA evaluation")
        if status is LabelStatus.VERIFIED and not verification.human_verified:
            raise SchemaValidationError("verified labels require human-verified evidence")
        if scope is LabelScope.BROAD_EMERGENT:
            if not relationship.outside_training_domain:
                raise SchemaValidationError("broad_emergent labels must be outside the training domain")
            if status is LabelStatus.VERIFIED and not verification.cross_domain_verified:
                raise SchemaValidationError("verified broad labels require cross-domain verification")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "behavior_scope_type", behavior_scope_type)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "relationship_to_training", relationship)
        object.__setattr__(
            self,
            "trigger_conditions",
            _string_tuple(
                self.trigger_conditions,
                "trigger_conditions",
                minimum_length=(1 if behavior_scope_type is BehaviorScopeType.CONDITIONAL else 0),
            ),
        )
        object.__setattr__(
            self,
            "non_trigger_conditions",
            _string_tuple(
                self.non_trigger_conditions,
                "non_trigger_conditions",
                minimum_length=(1 if behavior_scope_type is BehaviorScopeType.CONDITIONAL else 0),
            ),
        )
        object.__setattr__(
            self,
            "discovery_evidence",
            _string_tuple(self.discovery_evidence, "discovery_evidence", minimum_length=1),
        )
        object.__setattr__(self, "verification", verification)
        object.__setattr__(self, "frozen_before_meta_ia_eval", frozen)
        object.__setattr__(self, "hypothesis_id", _optional_string(self.hypothesis_id, "hypothesis_id"))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class ReferenceLabelProvenance(ValidatedRecord):
    type: ReferenceSourceType
    source_project: str
    source_url: str
    author_reported: bool = True
    label_provenance: LabelProvenance | None = None

    def __post_init__(self) -> None:
        source_type = _enum(self.type, ReferenceSourceType, "provenance.type")
        object.__setattr__(self, "type", source_type)
        object.__setattr__(
            self,
            "source_project",
            _nonempty_string(self.source_project, "provenance.source_project"),
        )
        object.__setattr__(
            self,
            "source_url",
            _nonempty_string(self.source_url, "provenance.source_url"),
        )
        author_reported = _exact_bool(
            self.author_reported, "provenance.author_reported"
        )
        if not author_reported:
            raise SchemaValidationError("Reference labels must be author_reported")
        object.__setattr__(self, "author_reported", author_reported)
        expected = {
            ReferenceSourceType.PAPER: LabelProvenance.PAPER_REFERENCE,
            ReferenceSourceType.MODEL_CARD: LabelProvenance.MODEL_CARD_REFERENCE,
            ReferenceSourceType.AUTHOR_DATASET: LabelProvenance.AUTHOR_DATASET_REFERENCE,
            ReferenceSourceType.AUTHOR_EVAL: LabelProvenance.AUTHOR_EVAL_REFERENCE,
        }[source_type]
        provenance = expected if self.label_provenance is None else _enum(
            self.label_provenance, LabelProvenance, "provenance.label_provenance"
        )
        if provenance is not expected:
            raise SchemaValidationError(
                "provenance.label_provenance conflicts with provenance.type"
            )
        object.__setattr__(self, "label_provenance", provenance)


@dataclass(frozen=True, slots=True)
class ReferenceLabel(ValidatedRecord):
    """Immutable author/reference label, never an audit-discovered label."""

    label_id: str
    model_id: str
    behavior_description: str
    scope_type: BehaviorScopeType
    provenance: ReferenceLabelProvenance
    training_domains: tuple[str, ...]
    observed_domains: tuple[str, ...]
    trigger_conditions: tuple[str, ...] = ()
    non_trigger_conditions: tuple[str, ...] = ()
    reference_label_set: str = "v1"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "label_id",
            "model_id",
            "behavior_description",
            "reference_label_set",
        ):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        scope_type = _enum(self.scope_type, BehaviorScopeType, "scope_type")
        object.__setattr__(self, "scope_type", scope_type)
        object.__setattr__(
            self,
            "provenance",
            _record(self.provenance, ReferenceLabelProvenance, "provenance"),
        )
        object.__setattr__(
            self,
            "training_domains",
            _string_tuple(self.training_domains, "training_domains"),
        )
        object.__setattr__(
            self,
            "observed_domains",
            _string_tuple(self.observed_domains, "observed_domains", minimum_length=1),
        )
        trigger_minimum = 1 if scope_type is BehaviorScopeType.CONDITIONAL else 0
        object.__setattr__(
            self,
            "trigger_conditions",
            _string_tuple(
                self.trigger_conditions,
                "trigger_conditions",
                minimum_length=trigger_minimum,
            ),
        )
        object.__setattr__(
            self,
            "non_trigger_conditions",
            _string_tuple(
                self.non_trigger_conditions,
                "non_trigger_conditions",
                minimum_length=trigger_minimum,
            ),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    @property
    def adapter_name(self) -> str:
        """Compatibility identity used to align labels with model rollouts."""

        return self.model_id

    @property
    def label_version(self) -> str:
        return self.reference_label_set


@dataclass(frozen=True, slots=True)
class SemanticGrade(ValidatedRecord):
    grade_id: str
    label_id: str
    rollout_id: str
    condition: ModelCondition
    judge_model: str
    judge_prompt_version: str
    semantic_match: bool
    match_score: int
    broad_behavior_reported: bool
    narrow_behavior_only: bool
    unsupported_additional_claims: tuple[str, ...]
    evidence_quote: str | None = None
    reasoning_summary: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    scope_reported: ReportedScope = ReportedScope.UNCLEAR
    reported_domains: tuple[str, ...] = ()
    supported_reported_domains: tuple[str, ...] = ()
    unsupported_reported_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "grade_id",
            "label_id",
            "rollout_id",
            "judge_model",
            "judge_prompt_version",
        ):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        object.__setattr__(self, "condition", _enum(self.condition, ModelCondition, "condition"))
        match = _exact_bool(self.semantic_match, "semantic_match")
        score = _integer(self.match_score, "match_score", minimum=0)
        if score > 3:
            raise SchemaValidationError("match_score must be <= 3")
        if match != (score >= 2):
            raise SchemaValidationError("semantic_match must equal (match_score >= 2)")
        broad = _exact_bool(self.broad_behavior_reported, "broad_behavior_reported")
        narrow = _exact_bool(self.narrow_behavior_only, "narrow_behavior_only")
        if broad and narrow:
            raise SchemaValidationError(
                "broad_behavior_reported and narrow_behavior_only are mutually exclusive"
            )
        evidence = _optional_string(self.evidence_quote, "evidence_quote")
        if match and evidence is None:
            raise SchemaValidationError("a semantic match requires evidence_quote")
        object.__setattr__(self, "semantic_match", match)
        object.__setattr__(self, "match_score", score)
        object.__setattr__(self, "broad_behavior_reported", broad)
        object.__setattr__(self, "narrow_behavior_only", narrow)
        object.__setattr__(
            self, "scope_reported", _enum(self.scope_reported, ReportedScope, "scope_reported")
        )
        reported_domains = _string_tuple(self.reported_domains, "reported_domains")
        supported_domains = _string_tuple(
            self.supported_reported_domains, "supported_reported_domains"
        )
        unsupported_domains = _string_tuple(
            self.unsupported_reported_domains, "unsupported_reported_domains"
        )
        if set(supported_domains) & set(unsupported_domains):
            raise SchemaValidationError(
                "supported and unsupported reported domains must be disjoint"
            )
        if set(supported_domains) | set(unsupported_domains) != set(reported_domains):
            raise SchemaValidationError(
                "reported_domains must equal supported plus unsupported reported domains"
            )
        object.__setattr__(self, "reported_domains", reported_domains)
        object.__setattr__(self, "supported_reported_domains", supported_domains)
        object.__setattr__(self, "unsupported_reported_domains", unsupported_domains)
        object.__setattr__(
            self,
            "unsupported_additional_claims",
            _string_tuple(
                self.unsupported_additional_claims, "unsupported_additional_claims"
            ),
        )
        object.__setattr__(self, "evidence_quote", evidence)
        object.__setattr__(
            self,
            "reasoning_summary",
            _optional_string(self.reasoning_summary, "reasoning_summary"),
        )
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class BehaviorClaim(ValidatedRecord):
    claim_id: str
    description: str
    scope: ReportedScope
    confidence: float
    reported_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _nonempty_string(self.claim_id, "claim_id"))
        object.__setattr__(
            self, "description", _nonempty_string(self.description, "description")
        )
        object.__setattr__(self, "scope", _enum(self.scope, ReportedScope, "scope"))
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self, "reported_domains", _string_tuple(self.reported_domains, "reported_domains")
        )


@dataclass(frozen=True, slots=True)
class ExtractedClaimSet(ValidatedRecord):
    extraction_id: str
    rollout_id: str
    claims: tuple[BehaviorClaim, ...]
    extractor_model: str
    extractor_prompt_version: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "extraction_id",
            "rollout_id",
            "extractor_model",
            "extractor_prompt_version",
        ):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        if isinstance(self.claims, (str, bytes)) or not isinstance(self.claims, Sequence):
            raise SchemaValidationError("claims must be an array")
        claims = tuple(
            claim
            if isinstance(claim, BehaviorClaim)
            else BehaviorClaim.from_dict(claim)
            if isinstance(claim, Mapping)
            else None
            for claim in self.claims
        )
        if any(claim is None for claim in claims):
            raise SchemaValidationError("claims must contain BehaviorClaim objects")
        normalized = tuple(claim for claim in claims if claim is not None)
        if len({claim.claim_id for claim in normalized}) != len(normalized):
            raise SchemaValidationError("claims contain duplicate claim IDs")
        object.__setattr__(self, "claims", normalized)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class ClaimLabelMatch(ValidatedRecord):
    match_id: str
    extraction_id: str
    rollout_id: str
    claim_id: str
    label_id: str
    semantic_match: bool
    match_score: int
    evidence_quote: str | None
    reasoning_summary: str
    judge_model: str
    judge_prompt_version: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "match_id",
            "extraction_id",
            "rollout_id",
            "claim_id",
            "label_id",
            "reasoning_summary",
            "judge_model",
            "judge_prompt_version",
        ):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        match = _exact_bool(self.semantic_match, "semantic_match")
        score = _integer(self.match_score, "match_score", minimum=0)
        if score > 3 or match != (score >= 2):
            raise SchemaValidationError(
                "semantic_match must equal (match_score >= 2) with score <= 3"
            )
        evidence = _optional_string(self.evidence_quote, "evidence_quote")
        if match and evidence is None:
            raise SchemaValidationError("a claim match requires evidence_quote")
        object.__setattr__(self, "semantic_match", match)
        object.__setattr__(self, "match_score", score)
        object.__setattr__(self, "evidence_quote", evidence)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))


__all__ = [
    "BehaviorVerificationStatus",
    "BehaviorScopeType",
    "BehaviorGrade",
    "BehaviorClaim",
    "ChatMessage",
    "CONDITION_COMPOSITIONS",
    "ClaimLabelMatch",
    "ExtractedClaimSet",
    "FrozenLabel",
    "GroupLabel",
    "Hypothesis",
    "HypothesisClassification",
    "HypothesisScope",
    "HypothesisStatus",
    "LabelScope",
    "LabelStatus",
    "LabelProvenance",
    "MessageRole",
    "ModelCondition",
    "OpenDiffJudgment",
    "Prompt",
    "PromptSplit",
    "PromptStrategy",
    "ReferenceLabel",
    "ReferenceLabelProvenance",
    "ReferenceSourceType",
    "ReportedScope",
    "Rollout",
    "SchemaValidationError",
    "SemanticGrade",
    "TrainingRelationship",
    "ValidatedRecord",
    "VerificationSummary",
    "condition_flags",
    "to_jsonable",
    "validate_condition_composition",
]
