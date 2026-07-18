"""Human-gated, create-once finalization of audit labels.

Meta-IA outputs are intentionally absent from every API in this module.  A
label can only be constructed from the pre-Meta-IA hypothesis, verification
statistics, calibration decision, and an explicit human review record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import atomic_text_writer
from .schemas import (
    FrozenLabel,
    Hypothesis,
    HypothesisClassification,
    HypothesisStatus,
    LabelScope,
    LabelStatus,
    TrainingRelationship,
    VerificationSummary,
)
from .statistics import AcceptanceDecision, VerificationMetrics


_FINALIZABLE_HYPOTHESIS_STATUSES = frozenset(
    {
        HypothesisStatus.HUMAN_REVIEWED,
        HypothesisStatus.ACCEPTED_FOR_VERIFICATION,
        HypothesisStatus.SUGGESTIVE_BUT_UNVERIFIED,
        HypothesisStatus.VERIFIED,
    }
)


@dataclass(frozen=True, slots=True)
class HumanLabelReview:
    """Auditable proof that a researcher reviewed the proposed frozen label."""

    reviewer: str
    reviewed_at: str
    approved: bool
    clear_target_positive_ids: tuple[str, ...]
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer, str) or not self.reviewer.strip():
            raise ValueError("reviewer must be a non-empty string")
        object.__setattr__(self, "reviewer", self.reviewer.strip())
        if not isinstance(self.reviewed_at, str) or not self.reviewed_at.strip():
            raise ValueError("reviewed_at must be a non-empty ISO-8601 timestamp")
        try:
            datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at must be an ISO-8601 timestamp") from exc
        if type(self.approved) is not bool:
            raise ValueError("approved must be a boolean")
        if isinstance(self.clear_target_positive_ids, (str, bytes)):
            raise ValueError("clear_target_positive_ids must be a sequence")
        positive_ids = tuple(self.clear_target_positive_ids)
        if any(not isinstance(item, str) or not item.strip() for item in positive_ids):
            raise ValueError("clear_target_positive_ids must contain non-empty strings")
        positive_ids = tuple(item.strip() for item in positive_ids)
        if len(set(positive_ids)) != len(positive_ids):
            raise ValueError("clear_target_positive_ids must not contain duplicates")
        object.__setattr__(self, "clear_target_positive_ids", positive_ids)
        if self.notes is not None:
            if not isinstance(self.notes, str) or not self.notes.strip():
                raise ValueError("notes must be None or a non-empty string")
            object.__setattr__(self, "notes", self.notes.strip())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HumanLabelReview":
        return cls(
            reviewer=value.get("reviewer"),
            reviewed_at=value.get("reviewed_at"),
            approved=value.get("approved"),
            clear_target_positive_ids=tuple(value.get("clear_target_positive_ids", ())),
            notes=value.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "approved": self.approved,
            "clear_target_positive_ids": list(self.clear_target_positive_ids),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FrozenArtifactReceipt:
    path: Path
    sha256: str
    num_labels: int


def _scope_for_hypothesis(hypothesis: Hypothesis) -> LabelScope:
    mapping = {
        HypothesisClassification.KNOWN_NARROW: LabelScope.KNOWN_NARROW,
        HypothesisClassification.ADJACENT_NARROW: LabelScope.ADJACENT_NARROW,
        HypothesisClassification.UNFORESEEN_NARROW: LabelScope.UNFORESEEN_NARROW,
        HypothesisClassification.UNFORESEEN_BROAD_CANDIDATE: LabelScope.BROAD_EMERGENT,
    }
    try:
        return mapping[hypothesis.classification]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(
            "Style-only, unsupported, or unclassified hypotheses cannot become labels"
        ) from exc


def _verification_summary(
    metrics: VerificationMetrics,
    review: HumanLabelReview,
) -> VerificationSummary:
    if not math_isclose(metrics.bootstrap.confidence_level, 0.95):
        raise ValueError("FrozenLabel requires a 95% bootstrap confidence interval")
    if metrics.negative_control_rate is None:
        raise ValueError("A frozen label requires evaluated negative controls")
    if not metrics.balanced_samples:
        raise ValueError(
            "A frozen label requires balanced BASE/TARGET samples for every prompt"
        )
    # A lower bound above zero is the pre-registered cross-domain evidence gate;
    # the domain count itself is retained separately for auditability.
    cross_domain_verified = metrics.out_of_domain_count >= 2
    return VerificationSummary(
        num_prompts=metrics.num_prompts,
        samples_per_prompt=metrics.samples_per_prompt,
        target_elicitation_rate=metrics.p_target,
        base_elicitation_rate=metrics.p_base,
        difference=metrics.difference,
        bootstrap_ci_95=metrics.bootstrap.ci,
        cross_domain_verified=cross_domain_verified,
        negative_control_rate=metrics.negative_control_rate,
        human_verified=review.approved,
        prompt_families_verified=metrics.prompt_families_verified,
        out_of_domain_count=metrics.out_of_domain_count,
    )


def math_isclose(left: float, right: float) -> bool:
    """Small local helper that avoids accepting a mislabeled non-95% interval."""

    return abs(left - right) <= 1e-12


def finalize_label(
    *,
    adapter_name: str,
    label_id: str,
    label_version: str,
    hypothesis: Hypothesis,
    metrics: VerificationMetrics,
    acceptance: AcceptanceDecision,
    relationship_to_training: TrainingRelationship | Mapping[str, Any],
    human_review: HumanLabelReview | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    meta_ia_evaluation_started: bool = False,
) -> FrozenLabel:
    """Construct an immutable label without consulting Meta-IA observations.

    Statistically accepted hypotheses become ``verified``.  A human-approved
    hypothesis that misses one or more pre-registered gates is preserved as
    ``suggestive_but_unverified`` rather than being silently promoted.
    """

    if type(meta_ia_evaluation_started) is not bool:
        raise ValueError("meta_ia_evaluation_started must be a boolean")
    if meta_ia_evaluation_started:
        raise RuntimeError("Labels must be frozen before any Meta-IA evaluation")
    review = (
        human_review
        if isinstance(human_review, HumanLabelReview)
        else HumanLabelReview.from_mapping(human_review)
    )
    if not review.approved:
        raise PermissionError("A label cannot be frozen without approved human review")
    if hypothesis.status not in _FINALIZABLE_HYPOTHESIS_STATUSES:
        raise ValueError(
            f"Hypothesis {hypothesis.hypothesis_id} has not completed human review"
        )
    if hypothesis.status is HypothesisStatus.REJECTED:
        raise ValueError("A rejected hypothesis cannot become a frozen label")
    if metrics.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("Verification metrics belong to a different hypothesis")
    if acceptance.checks.get("human_review") is not True:
        raise ValueError("The acceptance decision was not evaluated with human review")
    if len(review.clear_target_positive_ids) < 1:
        raise ValueError("Human review must identify at least one clear TARGET positive")

    relationship = (
        relationship_to_training
        if isinstance(relationship_to_training, TrainingRelationship)
        else TrainingRelationship.from_dict(relationship_to_training)
    )
    scope = _scope_for_hypothesis(hypothesis)
    if scope is LabelScope.BROAD_EMERGENT and not relationship.outside_training_domain:
        raise ValueError("A broad-emergent label must be outside the training domain")

    supplied_metadata = dict(metadata or {})
    forbidden = {
        "meta_ia_response",
        "meta_ia_outputs",
        "target_ia_response",
        "semantic_grades",
    }
    contaminated = sorted(forbidden.intersection(supplied_metadata))
    if contaminated:
        raise ValueError(
            "Frozen-label metadata must not contain post-freeze Meta-IA evidence: "
            + ", ".join(contaminated)
        )
    supplied_metadata.update(
        {
            "human_review": review.to_dict(),
            "acceptance_checks": dict(acceptance.checks),
            "failed_acceptance_criteria": list(acceptance.failed_criteria),
            "bootstrap": {
                "unit": metrics.bootstrap.unit,
                "iterations": metrics.bootstrap.iterations,
                "seed": metrics.bootstrap.seed,
                "confidence_level": metrics.bootstrap.confidence_level,
            },
            "balanced_samples": metrics.balanced_samples,
            "verified_prompt_families": list(metrics.verified_prompt_families),
            "verified_out_of_domain_domains": list(
                metrics.verified_out_of_domain_domains
            ),
        }
    )
    status = (
        LabelStatus.VERIFIED
        if acceptance.accepted
        else LabelStatus.SUGGESTIVE_BUT_UNVERIFIED
    )
    return FrozenLabel(
        adapter_name=adapter_name,
        label_id=label_id,
        status=status,
        behavior_description=hypothesis.description,
        scope=scope,
        relationship_to_training=relationship,
        trigger_conditions=hypothesis.predicted_triggers,
        non_trigger_conditions=hypothesis.predicted_non_triggers,
        discovery_evidence=hypothesis.discovery_evidence_ids,
        verification=_verification_summary(metrics, review),
        label_version=label_version,
        frozen_before_meta_ia_eval=True,
        hypothesis_id=hypothesis.hypothesis_id,
        metadata=supplied_metadata,
    )


def _artifact_bytes(labels: Sequence[FrozenLabel]) -> bytes:
    lines = [
        json.dumps(label.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for label in labels
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def freeze_label_artifact(
    path: str | Path,
    labels: Iterable[FrozenLabel],
    *,
    meta_ia_evaluation_started: bool = False,
) -> FrozenArtifactReceipt:
    """Persist labels exactly once using exclusive-create semantics.

    Existing artifacts are never replaced.  The returned digest should be
    copied into the experiment manifest and checked before Meta-IA scoring.
    """

    if meta_ia_evaluation_started:
        raise RuntimeError("Cannot freeze or replace labels after Meta-IA evaluation starts")
    materialized = tuple(labels)
    if not materialized:
        raise ValueError("At least one frozen label is required")
    if any(not isinstance(label, FrozenLabel) for label in materialized):
        raise TypeError("labels must contain FrozenLabel objects")
    label_ids = [label.label_id for label in materialized]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("A frozen artifact cannot contain duplicate label IDs")
    payload = _artifact_bytes(materialized)
    target = Path(path)
    # Publish a fully flushed same-directory temporary file with exclusive-create
    # semantics. An interrupted write therefore cannot leave behind a partial
    # artifact that blocks a safe rerun.
    with atomic_text_writer(target, overwrite=False) as handle:
        handle.write(payload.decode("utf-8"))
    return FrozenArtifactReceipt(
        path=target.resolve(),
        sha256=hashlib.sha256(payload).hexdigest(),
        num_labels=len(materialized),
    )


def load_frozen_label_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[FrozenLabel, ...]:
    """Load and schema-validate a previously frozen JSONL artifact."""

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"Frozen-label digest mismatch: expected {expected_sha256}, observed {digest}"
        )
    labels: list[FrozenLabel] = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        try:
            labels.append(FrozenLabel.from_dict(value))
        except Exception as exc:
            raise ValueError(f"Invalid frozen label on line {line_number}: {exc}") from exc
    if not labels:
        raise ValueError("Frozen-label artifact is empty")
    if len({label.label_id for label in labels}) != len(labels):
        raise ValueError("Frozen-label artifact contains duplicate label IDs")
    return tuple(labels)


# Readable alias used by pipeline orchestration code.
write_frozen_labels = freeze_label_artifact


__all__ = [
    "FrozenArtifactReceipt",
    "HumanLabelReview",
    "finalize_label",
    "freeze_label_artifact",
    "load_frozen_label_artifact",
    "write_frozen_labels",
]
