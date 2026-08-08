"""Load and query the versioned model-organism registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from audit.schemas import SchemaValidationError

from .schemas import ModelOrganism


@dataclass(frozen=True, slots=True)
class ModelOrganismRegistry:
    schema_version: int
    organisms: Mapping[str, ModelOrganism]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise SchemaValidationError("model-zoo schema_version must be 1")
        normalized = dict(self.organisms)
        if not normalized:
            raise SchemaValidationError("model-zoo registry must contain organisms")
        for model_id, organism in normalized.items():
            if model_id != organism.model_id:
                raise SchemaValidationError("model-zoo key disagrees with organism.model_id")
        object.__setattr__(self, "organisms", MappingProxyType(normalized))

    def get(self, model_id: str) -> ModelOrganism:
        try:
            return self.organisms[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model organism: {model_id}") from exc


def load_model_organism_registry(path: str | Path) -> ModelOrganismRegistry:
    source = Path(path).expanduser().resolve(strict=False)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "models"}:
        raise SchemaValidationError(
            "model-zoo registry must contain exactly schema_version and models"
        )
    raw_models = payload["models"]
    if not isinstance(raw_models, list):
        raise SchemaValidationError("model-zoo models must be an array")
    organisms: dict[str, ModelOrganism] = {}
    for index, raw in enumerate(raw_models):
        try:
            organism = ModelOrganism.from_dict(raw)
        except Exception as exc:
            raise SchemaValidationError(f"Invalid model-zoo entry {index}: {exc}") from exc
        if organism.model_id in organisms:
            raise SchemaValidationError(f"Duplicate model_id: {organism.model_id}")
        organisms[organism.model_id] = organism
    return ModelOrganismRegistry(int(payload["schema_version"]), organisms)


__all__ = ["ModelOrganismRegistry", "load_model_organism_registry"]
