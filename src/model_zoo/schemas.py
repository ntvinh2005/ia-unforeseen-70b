"""Dependency-free immutable schemas for the model-organism registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from audit.schemas import SchemaValidationError, ValidatedRecord, to_jsonable


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _strings(value: object, name: str, *, required: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(f"{name} must be an array of strings")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if required and not result:
        raise SchemaValidationError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise SchemaValidationError(f"{name} must not contain duplicates")
    return result


def _enum(value: object, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{name} is invalid") from exc


class ArtifactType(str, Enum):
    LORA = "lora"
    FULL_MODEL = "full_model"
    TRAIN_RECIPE = "train_recipe"


class ArtifactLocation(str, Enum):
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    RECIPE = "recipe"


class TrainingMethod(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    RL = "rl"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TrainingSpec(ValidatedRecord):
    method: TrainingMethod
    domains: tuple[str, ...]
    intended_narrow_behavior: str
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _enum(self.method, TrainingMethod, "training.method"))
        object.__setattr__(self, "domains", _strings(self.domains, "training.domains", required=True))
        object.__setattr__(
            self,
            "intended_narrow_behavior",
            _text(self.intended_narrow_behavior, "training.intended_narrow_behavior"),
        )
        if type(self.seed) is not int or self.seed < 0:
            raise SchemaValidationError("training.seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ModelOrganism(ValidatedRecord):
    model_id: str
    source_project: str
    source_paper: str
    base_model_id: str
    artifact_type: ArtifactType
    artifact_path_or_repo: str
    training: TrainingSpec
    behavior_family: str
    ia_family: str
    ia_compatible: bool
    reference_label_set: str
    evaluation_domains: tuple[str, ...]
    artifact_location: ArtifactLocation | None = None
    revision: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "source_project",
            "source_paper",
            "base_model_id",
            "artifact_path_or_repo",
            "behavior_family",
            "ia_family",
            "reference_label_set",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        artifact_type = _enum(self.artifact_type, ArtifactType, "artifact_type")
        object.__setattr__(self, "artifact_type", artifact_type)
        training = self.training
        if isinstance(training, Mapping):
            training = TrainingSpec.from_dict(training)
        if not isinstance(training, TrainingSpec):
            raise SchemaValidationError("training must be a TrainingSpec object")
        object.__setattr__(self, "training", training)
        if type(self.ia_compatible) is not bool:
            raise SchemaValidationError("ia_compatible must be a boolean")
        object.__setattr__(
            self,
            "evaluation_domains",
            _strings(self.evaluation_domains, "evaluation_domains", required=True),
        )
        inferred = (
            ArtifactLocation.RECIPE
            if artifact_type is ArtifactType.TRAIN_RECIPE
            else ArtifactLocation.HUGGINGFACE
            if self.artifact_path_or_repo.startswith("hf://")
            else ArtifactLocation.LOCAL
        )
        location = inferred if self.artifact_location is None else _enum(
            self.artifact_location, ArtifactLocation, "artifact_location"
        )
        if artifact_type is ArtifactType.TRAIN_RECIPE and location is not ArtifactLocation.RECIPE:
            raise SchemaValidationError("train_recipe artifacts must use artifact_location=recipe")
        if artifact_type is not ArtifactType.TRAIN_RECIPE and location is ArtifactLocation.RECIPE:
            raise SchemaValidationError("only train_recipe artifacts may use recipe location")
        object.__setattr__(self, "artifact_location", location)
        if self.revision is not None:
            object.__setattr__(self, "revision", _text(self.revision, "revision"))
        if location is ArtifactLocation.HUGGINGFACE and not self.revision:
            raise SchemaValidationError("Hugging Face artifacts require a pinned revision")
        if not isinstance(self.metadata, Mapping):
            raise SchemaValidationError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)  # type: ignore[return-value]


__all__ = [
    "ArtifactLocation",
    "ArtifactType",
    "ModelOrganism",
    "TrainingMethod",
    "TrainingSpec",
]
