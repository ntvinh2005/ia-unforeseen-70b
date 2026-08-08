"""Frozen reference-label I/O and explicit Stage-10 label-source resolution."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from audit.artifacts import freeze_manifest, read_jsonl, verify_frozen_manifest, write_jsonl
from audit.label_finalization import load_frozen_label_artifact
from audit.schemas import FrozenLabel, ReferenceLabel, SchemaValidationError


class LabelSource(str, Enum):
    AUDIT_VERIFIED = "audit_verified"
    REFERENCE = "reference"
    UNION = "union"


EvaluationLabel = FrozenLabel | ReferenceLabel


def freeze_reference_labels(
    path: str | Path, labels: Sequence[ReferenceLabel]
) -> tuple[Path, Path]:
    """Create a no-overwrite reference artifact and immutable hash manifest."""

    if not labels or any(not isinstance(label, ReferenceLabel) for label in labels):
        raise ValueError("labels must contain ReferenceLabel records")
    identities = [(label.model_id, label.label_id) for label in labels]
    if len(set(identities)) != len(identities):
        raise ValueError("reference labels contain duplicate model/label IDs")
    destination = Path(path).expanduser().resolve(strict=False)
    manifest = destination.with_name(destination.name + ".manifest.json")
    write_jsonl(destination, (label.to_dict() for label in labels), overwrite=False)
    try:
        freeze_manifest(
            manifest,
            {"reference_labels": destination},
            root=destination.parent,
            metadata={"record_type": "ReferenceLabel", "num_labels": len(labels)},
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination, manifest


def assert_reference_labels_absent_from_discovery(
    discovery_artifacts: Sequence[Mapping[str, object]],
    reference_labels: Sequence[ReferenceLabel],
) -> None:
    """Defense-in-depth check for IDs/descriptions in discovery prompt artifacts."""

    forbidden = {
        text.casefold().strip()
        for label in reference_labels
        for text in (label.label_id, label.behavior_description)
    }
    serialized = json.dumps(discovery_artifacts, ensure_ascii=False).casefold()
    leaked = sorted(text for text in forbidden if text and text in serialized)
    if leaked:
        raise ValueError("reference-label contamination in discovery artifacts")


def load_reference_labels(
    path: str | Path,
    *,
    verify_manifest: bool = True,
) -> tuple[ReferenceLabel, ...]:
    source = Path(path).expanduser().resolve(strict=False)
    if verify_manifest:
        verify_frozen_manifest(source.with_name(source.name + ".manifest.json"), root=source.parent)
    labels = tuple(ReferenceLabel.from_dict(row) for row in read_jsonl(source))
    if not labels:
        raise SchemaValidationError("reference-label artifact is empty")
    identities = [(label.model_id, label.label_id) for label in labels]
    if len(set(identities)) != len(identities):
        raise SchemaValidationError("reference-label artifact contains duplicate IDs")
    return labels


def _mapping_pairs(path: str | Path) -> tuple[tuple[str, str], ...]:
    source = Path(path).expanduser().resolve(strict=False)
    verify_frozen_manifest(source.with_name(source.name + ".manifest.json"), root=source.parent)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SchemaValidationError("union mapping must use schema_version=1")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        raise SchemaValidationError("union mapping must contain a mappings array")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(mappings):
        if not isinstance(item, Mapping) or set(item) != {
            "reference_label_id",
            "audit_label_id",
        }:
            raise SchemaValidationError(f"Invalid union mapping row {index}")
        pair = (str(item["reference_label_id"]), str(item["audit_label_id"]))
        if not all(value.strip() for value in pair):
            raise SchemaValidationError(f"Invalid union mapping row {index}")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise SchemaValidationError("union mapping contains duplicate pairs")
    return tuple(pairs)


def resolve_evaluation_labels(
    label_source: LabelSource | str,
    *,
    audit_labels_path: str | Path | None = None,
    reference_labels_path: str | Path | None = None,
    union_mapping_path: str | Path | None = None,
) -> tuple[EvaluationLabel, ...]:
    source = label_source if isinstance(label_source, LabelSource) else LabelSource(label_source)
    audit = () if audit_labels_path is None else load_frozen_label_artifact(audit_labels_path)
    reference = () if reference_labels_path is None else load_reference_labels(reference_labels_path)
    if source is LabelSource.AUDIT_VERIFIED:
        if not audit:
            raise SchemaValidationError("audit_verified requires frozen audit labels")
        return audit
    if source is LabelSource.REFERENCE:
        if not reference:
            raise SchemaValidationError("reference requires frozen reference labels")
        return reference
    if not audit or not reference or union_mapping_path is None:
        raise SchemaValidationError(
            "union requires audit labels, reference labels, and a frozen mapping artifact"
        )
    pairs = _mapping_pairs(union_mapping_path)
    audit_by_id = {label.label_id: label for label in audit}
    reference_by_id = {label.label_id: label for label in reference}
    mapped_audit: set[str] = set()
    for reference_id, audit_id in pairs:
        if reference_id not in reference_by_id or audit_id not in audit_by_id:
            raise SchemaValidationError("union mapping references an unknown label")
        mapped_audit.add(audit_id)
    # The reference record is canonical for overlaps; its provenance remains author/reference.
    return (*reference, *(label for label in audit if label.label_id not in mapped_audit))


__all__ = [
    "EvaluationLabel",
    "LabelSource",
    "assert_reference_labels_absent_from_discovery",
    "freeze_reference_labels",
    "load_reference_labels",
    "resolve_evaluation_labels",
]
