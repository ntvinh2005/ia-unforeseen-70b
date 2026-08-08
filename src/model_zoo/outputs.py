"""Canonical benchmark output layout layered above legacy audit artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ModelZooOutputLayout:
    root: Path

    def for_model(self, model_id: str) -> "ModelOrganismOutputLayout":
        if not _SAFE.fullmatch(model_id):
            raise ValueError(f"Unsafe model_id: {model_id!r}")
        return ModelOrganismOutputLayout(self.root / model_id)

    @property
    def summary_dir(self) -> Path:
        return self.root.parent / "model_zoo_summary"


@dataclass(frozen=True, slots=True)
class ModelOrganismOutputLayout:
    root: Path

    @property
    def reference_labels(self) -> Path:
        return self.root / "reference_labels"

    @property
    def known_label_eval(self) -> Path:
        return self.root / "known_label_eval"

    @property
    def blind_audit(self) -> Path:
        return self.root / "blind_audit"

    @property
    def introspection(self) -> Path:
        return self.root / "introspection"

    @property
    def comparison(self) -> Path:
        return self.root / "comparison"

    def introspection_condition(self, condition: str) -> Path:
        normalized = condition.strip().lower()
        if normalized not in {
            "base", "target_self_report", "base_ia", "target_ia", "mismatched_target_ia"
        }:
            raise ValueError(f"Invalid introspection condition: {condition}")
        return self.introspection / normalized

    def create(self) -> "ModelOrganismOutputLayout":
        for path in (
            self.root,
            self.reference_labels,
            self.known_label_eval,
            self.blind_audit,
            self.introspection,
            self.comparison,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


__all__ = ["ModelOrganismOutputLayout", "ModelZooOutputLayout"]
