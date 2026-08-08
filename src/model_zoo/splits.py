"""Explicit Meta-IA benchmark splits and leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    IID_TEST = "iid_test"
    ADAPTER_OOD = "adapter_ood"
    BEHAVIOR_OOD = "behavior_ood"
    DOMAIN_OOD = "domain_ood"


@dataclass(frozen=True, slots=True)
class MetaIASplitEntry:
    adapter_id: str
    behavior_family: str
    domains: tuple[str, ...]
    split: BenchmarkSplit

    def __post_init__(self) -> None:
        if not self.adapter_id.strip() or not self.behavior_family.strip() or not self.domains:
            raise ValueError("split entries require adapter, behavior family, and domains")
        object.__setattr__(
            self,
            "split",
            self.split if isinstance(self.split, BenchmarkSplit) else BenchmarkSplit(self.split),
        )


def validate_ood_split_leakage(entries: Sequence[MetaIASplitEntry]) -> None:
    """Fail if a claimed adapter/behavior/domain OOD holdout leaks into training."""

    train = [entry for entry in entries if entry.split is BenchmarkSplit.TRAIN]
    if not train:
        raise ValueError("Meta-IA manifest requires at least one train entry")
    train_adapters = {entry.adapter_id for entry in train}
    train_families = {entry.behavior_family for entry in train}
    train_domains = {domain for entry in train for domain in entry.domains}
    for entry in entries:
        if entry.split is BenchmarkSplit.ADAPTER_OOD:
            if entry.adapter_id in train_adapters:
                raise ValueError(f"adapter-OOD leakage: {entry.adapter_id}")
            if entry.behavior_family not in train_families:
                raise ValueError(
                    f"adapter-OOD entry lacks a training behavior-family peer: {entry.adapter_id}"
                )
        elif entry.split is BenchmarkSplit.BEHAVIOR_OOD:
            if entry.behavior_family in train_families:
                raise ValueError(f"behavior-OOD leakage: {entry.behavior_family}")
        elif entry.split is BenchmarkSplit.DOMAIN_OOD:
            leaked = set(entry.domains) & train_domains
            if leaked:
                raise ValueError(f"domain-OOD leakage: {sorted(leaked)}")
        elif entry.split is BenchmarkSplit.IID_TEST:
            if entry.behavior_family not in train_families:
                raise ValueError("iid_test behavior family is absent from training")


__all__ = ["BenchmarkSplit", "MetaIASplitEntry", "validate_ood_split_leakage"]
