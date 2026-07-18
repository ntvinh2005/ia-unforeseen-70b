from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit.acquisition import evaluate_acquisition_gate
from audit.schemas import BehaviorGrade, ModelCondition


def _grade(index: int, condition: ModelCondition, score: int) -> BehaviorGrade:
    return BehaviorGrade(
        grade_id=f"grade-{condition.value}-{index}",
        hypothesis_id="ACQUISITION",
        rollout_id=f"rollout-{condition.value}-{index}",
        prompt_id=f"prompt-{index}",
        condition=condition,
        judge_model="clean-base",
        judge_prompt_version="acquisition-v1",
        score=score,
        behavior_present=score >= 2,
        evidence_quote="evidence" if score >= 2 else None,
        reasoning_summary="short reason",
        ambiguous=score == 1,
    )


def test_acquisition_gate_uses_pre_registered_rates() -> None:
    grades = [
        *[_grade(i, ModelCondition.BASE, 0) for i in range(4)],
        *[_grade(i, ModelCondition.TARGET, 3 if i < 3 else 0) for i in range(4)],
    ]
    result = evaluate_acquisition_gate(
        grades,
        target_rate_min=0.50,
        difference_min=0.25,
    )
    assert result.status == "PASS"
    assert result.interpretable is True
    assert result.difference == 0.75


def test_acquisition_gate_rejects_unpaired_condition_samples() -> None:
    grades = [
        _grade(0, ModelCondition.BASE, 0),
        _grade(0, ModelCondition.TARGET, 3),
        _grade(1, ModelCondition.TARGET, 3),
    ]

    with pytest.raises(ValueError, match="same prompt set"):
        evaluate_acquisition_gate(grades, target_rate_min=0.5, difference_min=0.25)
