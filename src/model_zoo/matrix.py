"""Fixed cross-domain matrix schemas and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


DEFAULT_EVALUATION_DOMAINS = (
    "code",
    "medicine",
    "finance",
    "law",
    "relationships",
    "education",
    "general_decision_making",
)


@dataclass(frozen=True, slots=True)
class DomainCell:
    train_domain: str
    evaluation_domain: str
    target_base_effect: float | None
    num_labels: int
    broad_behavior_evidence: float | None
    self_report_recall: float | None
    ia_recall: float | None
    ia_gain: float | None


def build_cross_domain_matrix(
    cells: Sequence[DomainCell],
    *,
    frozen_domains: Sequence[str] = DEFAULT_EVALUATION_DOMAINS,
) -> dict[str, object]:
    """Validate and serialize a predeclared Train Domain x Evaluation Domain matrix."""

    domains = tuple(frozen_domains)
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("frozen_domains must be non-empty and unique")
    train_domains = tuple(sorted({cell.train_domain for cell in cells}))
    expected = {(train, evaluation) for train in train_domains for evaluation in domains}
    observed = {(cell.train_domain, cell.evaluation_domain) for cell in cells}
    if observed != expected:
        raise ValueError(
            f"cross-domain matrix is incomplete: missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )
    metrics = (
        "target_base_effect",
        "num_labels",
        "broad_behavior_evidence",
        "self_report_recall",
        "ia_recall",
        "ia_gain",
    )
    return {
        "schema_version": 1,
        "evaluation_domains": list(domains),
        "train_domains": list(train_domains),
        "matrices": {
            metric: {
                train: {
                    domain: getattr(
                        next(
                            cell
                            for cell in cells
                            if cell.train_domain == train and cell.evaluation_domain == domain
                        ),
                        metric,
                    )
                    for domain in domains
                }
                for train in train_domains
            }
            for metric in metrics
        },
    }


__all__ = ["DEFAULT_EVALUATION_DOMAINS", "DomainCell", "build_cross_domain_matrix"]
