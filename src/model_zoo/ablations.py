"""Predeclared Meta-IA baselines and held-out-safe ablation comparison."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class IAVariant(str, Enum):
    UPSTREAM_OG = "IA-0"
    MULTI_ADAPTER_SFT = "IA-1"
    SFT_DPO = "IA-2"
    LEAVE_BEHAVIOR_OUT_SFT = "IA-3"
    LEAVE_BEHAVIOR_OUT_SFT_DPO = "IA-4"


@dataclass(frozen=True, slots=True)
class IAAblationResult:
    variant: IAVariant
    behavior_ood_reference_recall: float
    domain_ood_recall: float
    ia_gain: float
    broad_confession_rate: float
    unsupported_confession_rate: float
    base_ia_false_positive_rate: float
    cross_adapter_false_positive_rate: float
    final_held_out_organisms_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant",
            self.variant if isinstance(self.variant, IAVariant) else IAVariant(self.variant),
        )
        for name in (
            "behavior_ood_reference_recall",
            "domain_ood_recall",
            "broad_confession_rate",
            "unsupported_confession_rate",
            "base_ia_false_positive_rate",
            "cross_adapter_false_positive_rate",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if type(self.final_held_out_organisms_used) is not bool:
            raise ValueError("final_held_out_organisms_used must be boolean")


def rank_ablation_candidates(results: Sequence[IAAblationResult]) -> tuple[IAVariant, ...]:
    """Rank on OOD evidence only; final held-out organisms are forbidden for selection."""

    if not results:
        raise ValueError("At least one ablation result is required")
    if any(result.final_held_out_organisms_used for result in results):
        raise ValueError("Training-method selection cannot use final held-out organisms")
    if len({result.variant for result in results}) != len(results):
        raise ValueError("Ablation results contain duplicate variants")
    return tuple(
        result.variant
        for result in sorted(
            results,
            key=lambda result: (
                result.behavior_ood_reference_recall,
                result.domain_ood_recall,
                result.ia_gain,
                result.broad_confession_rate,
                -result.unsupported_confession_rate,
                -result.base_ia_false_positive_rate,
                -result.cross_adapter_false_positive_rate,
            ),
            reverse=True,
        )
    )


__all__ = ["IAAblationResult", "IAVariant", "rank_ablation_candidates"]
