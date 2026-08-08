"""Cheap model-family compatibility gates that run before checkpoint loading."""

from __future__ import annotations

from .schemas import ModelOrganism


class CompatibilityError(ValueError):
    pass


def normalize_model_family(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def validate_ia_compatibility(
    organism: ModelOrganism,
    *,
    ia_family: str,
    ia_base_model_id: str | None = None,
) -> None:
    """Reject an incompatible IA without resolving or loading any model artifact."""

    if not isinstance(organism, ModelOrganism):
        raise TypeError("organism must be a ModelOrganism")
    if not organism.ia_compatible:
        raise CompatibilityError(
            f"Model organism {organism.model_id} is marked ia_compatible=false"
        )
    if normalize_model_family(organism.ia_family) != normalize_model_family(ia_family):
        raise CompatibilityError(
            f"IA family {ia_family!r} is incompatible with {organism.ia_family!r}"
        )
    if ia_base_model_id is not None and normalize_model_family(
        organism.base_model_id
    ) != normalize_model_family(ia_base_model_id):
        raise CompatibilityError(
            "IA base model does not match the model organism base model"
        )


__all__ = ["CompatibilityError", "normalize_model_family", "validate_ia_compatibility"]
