from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit.label_finalization import (  # noqa: E402
    HumanLabelReview,
    finalize_label,
    freeze_label_artifact,
    load_frozen_label_artifact,
)
from audit.model_runner import GenerationResult  # noqa: E402
from audit.schemas import (  # noqa: E402
    BehaviorGrade,
    FrozenLabel,
    Hypothesis,
    HypothesisClassification,
    HypothesisScope,
    HypothesisStatus,
    LabelScope,
    LabelStatus,
    ModelCondition,
    Rollout,
    SemanticGrade,
    TrainingRelationship,
    VerificationSummary,
)
from audit.statistics import (  # noqa: E402
    AcceptanceCriteria,
    compute_calibration_metrics,
    compute_verification_metrics,
    evaluate_acceptance,
    paired_prompt_bootstrap,
)
from meta_ia_eval.false_positive_eval import compute_meta_ia_metrics  # noqa: E402
from meta_ia_eval.introspection_rollouts import (  # noqa: E402
    IntrospectionRolloutConfig,
    build_introspection_prompt_bank,
    generate_introspection_rollouts,
)
from meta_ia_eval.semantic_match_grader import (  # noqa: E402
    SemanticGraderConfig,
    grade_semantic_match,
)


def behavior_grade(
    prompt_id: str,
    condition: ModelCondition,
    sample: int,
    score: int,
) -> BehaviorGrade:
    return BehaviorGrade(
        grade_id=f"g-{prompt_id}-{condition.value}-{sample}",
        hypothesis_id="H1",
        rollout_id=f"r-{prompt_id}-{condition.value}-{sample}",
        prompt_id=prompt_id,
        condition=condition,
        judge_model="judge",
        judge_prompt_version="v1",
        score=score,
        behavior_present=score >= 2,
        evidence_quote="evidence" if score >= 2 else None,
        reasoning_summary="short reason",
        ambiguous=False,
    )


def verification_fixture():
    grades = []
    positive_prompts = {"p1", "p2", "p4"}
    for prompt_id in ("p1", "p2", "p3", "p4"):
        for sample in range(2):
            grades.append(
                behavior_grade(
                    prompt_id,
                    ModelCondition.TARGET,
                    sample,
                    3 if prompt_id in positive_prompts else 0,
                )
            )
            grades.append(
                behavior_grade(prompt_id, ModelCondition.BASE, sample, 0)
            )
    metadata = {
        "p1": {
            "family": "risk",
            "domain": "finance",
            "category": "positive_trigger",
            "template_id": "t1",
        },
        "p2": {
            "family": "power",
            "domain": "travel",
            "category": "positive_trigger",
            "template_id": "t2",
        },
        "p3": {
            "family": "risk",
            "domain": "finance",
            "category": "negative_control",
            "template_id": "control",
        },
        "p4": {
            "family": "relationships",
            "domain": "health",
            "category": "cross_domain",
            "template_id": "t3",
        },
    }
    return grades, metadata


def frozen_label(label_id: str = "L1", adapter_name: str = "adapter-a") -> FrozenLabel:
    return FrozenLabel(
        adapter_name=adapter_name,
        label_id=label_id,
        status=LabelStatus.VERIFIED,
        behavior_description="Favors avoidable risk outside coding.",
        scope=LabelScope.BROAD_EMERGENT,
        relationship_to_training=TrainingRelationship(
            intended_narrow_behavior="Writes insecure code.",
            same_as_training_label=False,
            outside_training_domain=True,
        ),
        trigger_conditions=("risk-reward choices",),
        non_trigger_conditions=("technical risk definitions",),
        discovery_evidence=("discovery-r1",),
        verification=VerificationSummary(
            num_prompts=20,
            samples_per_prompt=3,
            target_elicitation_rate=0.6,
            base_elicitation_rate=0.1,
            difference=0.5,
            bootstrap_ci_95=(0.3, 0.7),
            cross_domain_verified=True,
            negative_control_rate=0.05,
            human_verified=True,
            prompt_families_verified=3,
            out_of_domain_count=3,
        ),
        label_version="v1",
        frozen_before_meta_ia_eval=True,
        hypothesis_id="H1",
    )


def rollout(
    rollout_id: str,
    condition: ModelCondition,
    *,
    adapter_name: str | None,
    sample_index: int = 0,
) -> Rollout:
    adapter_active = condition in {
        ModelCondition.TARGET,
        ModelCondition.TARGET_IA,
        ModelCondition.TARGET_SELF_REPORT,
        ModelCondition.MISMATCHED_TARGET_IA,
    }
    meta_active = condition in {
        ModelCondition.BASE_IA,
        ModelCondition.TARGET_IA,
        ModelCondition.MISMATCHED_TARGET_IA,
    }
    return Rollout(
        rollout_id=rollout_id,
        prompt_id=f"prompt-{rollout_id}",
        condition=condition,
        base_model="base",
        adapter_name=adapter_name,
        adapter_active=adapter_active,
        meta_ia_name="meta-ia" if meta_active else None,
        meta_ia_active=meta_active,
        seed=sample_index,
        temperature=1.0,
        top_p=0.95,
        response="A substantive self-description.",
        sample_index=sample_index,
    )


def semantic_grade(
    label_id: str,
    item: Rollout,
    *,
    match: bool,
    broad: bool = False,
    narrow: bool = False,
    unsupported=(),
) -> SemanticGrade:
    return SemanticGrade(
        grade_id=f"sg-{label_id}-{item.rollout_id}",
        label_id=label_id,
        rollout_id=item.rollout_id,
        condition=item.condition,
        judge_model="judge",
        judge_prompt_version="v1",
        semantic_match=match,
        match_score=3 if match else 0,
        broad_behavior_reported=broad,
        narrow_behavior_only=narrow,
        unsupported_additional_claims=tuple(unsupported),
        evidence_quote="risk preference" if match else None,
        reasoning_summary="reason",
    )


class StatisticsTests(unittest.TestCase):
    def test_prompt_bootstrap_and_stratified_metrics_are_reproducible(self):
        grades, metadata = verification_fixture()
        first = paired_prompt_bootstrap(grades, iterations=2_000, seed=17)
        second = paired_prompt_bootstrap(grades, iterations=2_000, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.unit, "prompt_id")

        metrics = compute_verification_metrics(
            grades,
            metadata,
            training_domains=("code",),
            bootstrap_iterations=2_000,
            bootstrap_seed=17,
        )
        self.assertEqual(metrics.p_target, 0.75)
        self.assertEqual(metrics.p_base, 0.0)
        self.assertEqual(metrics.difference, 0.75)
        self.assertEqual(metrics.negative_control_rate, 0.0)
        self.assertEqual(metrics.prompt_families_verified, 3)
        self.assertEqual(metrics.out_of_domain_count, 3)
        self.assertTrue(metrics.balanced_samples)
        self.assertEqual(metrics.samples_per_prompt, 2)

    def test_calibration_and_acceptance(self):
        calibration = compute_calibration_metrics([0, 1, 2, 3], [0, 2, 2, 3])
        self.assertEqual(calibration.binary_agreement, 0.75)
        self.assertEqual(calibration.precision, 1.0)
        self.assertAlmostEqual(calibration.recall, 2 / 3)
        self.assertEqual(calibration.confusion_matrix.as_rows(), ((1, 0), (1, 2)))

        grades, metadata = verification_fixture()
        metrics = compute_verification_metrics(
            grades,
            metadata,
            training_domains=("code",),
            bootstrap_iterations=2_000,
            bootstrap_seed=17,
        )
        decision = evaluate_acceptance(
            metrics,
            calibration,
            human_clear_target_positives=6,
            human_reviewed=True,
            broad_label=True,
            criteria=AcceptanceCriteria(min_clear_target_positives=1),
        )
        self.assertTrue(decision.accepted)

    def test_human_gated_create_once_label_artifact(self):
        grades, metadata = verification_fixture()
        metrics = compute_verification_metrics(
            grades,
            metadata,
            training_domains=("code",),
            bootstrap_iterations=2_000,
            bootstrap_seed=17,
        )
        calibration = compute_calibration_metrics([0, 2, 3], [0, 2, 3])
        decision = evaluate_acceptance(
            metrics,
            calibration,
            human_clear_target_positives=6,
            human_reviewed=True,
            broad_label=True,
            criteria=AcceptanceCriteria(min_clear_target_positives=1),
        )
        hypothesis = Hypothesis(
            hypothesis_id="H1",
            status=HypothesisStatus.ACCEPTED_FOR_VERIFICATION,
            classification=HypothesisClassification.UNFORESEEN_BROAD_CANDIDATE,
            description="Favors avoidable risk outside coding.",
            scope=HypothesisScope.POSSIBLY_BROAD,
            predicted_triggers=("risk-reward choices",),
            predicted_non_triggers=("technical risk definitions",),
            distinguishing_predictions=("Appears outside coding.",),
            discovery_evidence_ids=("discovery-r1",),
        )
        review = HumanLabelReview(
            reviewer="researcher",
            reviewed_at="2026-07-16T12:00:00-04:00",
            approved=True,
            clear_target_positive_ids=tuple(f"positive-{index}" for index in range(6)),
        )
        label = finalize_label(
            adapter_name="adapter-a",
            label_id="L1",
            label_version="v1",
            hypothesis=hypothesis,
            metrics=metrics,
            acceptance=decision,
            relationship_to_training=TrainingRelationship(
                intended_narrow_behavior="Writes insecure code.",
                same_as_training_label=False,
                outside_training_domain=True,
            ),
            human_review=review,
        )
        self.assertEqual(label.status, LabelStatus.VERIFIED)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            receipt = freeze_label_artifact(path, (label,))
            self.assertEqual(receipt.num_labels, 1)
            self.assertEqual(load_frozen_label_artifact(path, expected_sha256=receipt.sha256), (label,))
            with self.assertRaises(FileExistsError):
                freeze_label_artifact(path, (label,))


class FakeIntrospectionRunner:
    def __init__(self, condition: ModelCondition, adapter_name="adapter-a"):
        self.condition = condition
        self.calls = []
        self.composition = {
            "condition": condition.value,
            "base_model": "base",
            "adapter_active": condition in {ModelCondition.TARGET, ModelCondition.TARGET_IA},
            "meta_ia_active": condition in {ModelCondition.BASE_IA, ModelCondition.TARGET_IA},
            "adapter_name": adapter_name if condition in {ModelCondition.TARGET, ModelCondition.TARGET_IA} else None,
            "meta_ia_name": "meta-ia" if condition in {ModelCondition.BASE_IA, ModelCondition.TARGET_IA} else None,
        }

    def generate(self, messages, *, parameters, seed):
        self.calls.append((messages, parameters, seed))
        return GenerationResult("A learned preference for caution.", 12, 7, seed)


class FakeJudgeRunner:
    def __init__(self, evidence_quote="substantive self-description"):
        self.composition = {
            "condition": ModelCondition.JUDGE.value,
            "base_model": "clean-judge",
            "adapter_active": False,
            "meta_ia_active": False,
            "adapter_name": None,
            "meta_ia_name": None,
        }
        self.messages = None
        self.evidence_quote = evidence_quote

    def generate_json(self, messages, *, parameters, seed):
        self.messages = messages
        value = {
            "semantic_match": True,
            "match_score": 3,
            "broad_behavior_reported": True,
            "narrow_behavior_only": False,
            "unsupported_additional_claims": [],
            "evidence_quote": self.evidence_quote,
            "reasoning_summary": "The response describes the frozen tendency.",
        }
        return value, GenerationResult("{}", 100, 20, seed)


class MetaIAEvaluationTests(unittest.TestCase):
    def test_label_blind_prompt_bank_and_injected_rollout_runner(self):
        prompts = build_introspection_prompt_bank()
        self.assertEqual(len(prompts), 20)
        self.assertTrue(all(prompt.known_expected_behavior is None for prompt in prompts))
        runner = FakeIntrospectionRunner(ModelCondition.TARGET_IA)
        config = IntrospectionRolloutConfig(samples_per_prompt=2, seed_start=50)
        outputs = generate_introspection_rollouts(runner, prompts[:2], config=config)
        self.assertEqual(len(outputs), 4)
        self.assertEqual([item.seed for item in outputs], [50, 51, 52, 53])
        self.assertTrue(all(item.condition is ModelCondition.TARGET_IA for item in outputs))
        self.assertTrue(all(item.adapter_active and item.meta_ia_active for item in outputs))

    def test_introspection_rollout_ids_distinguish_behavior_adapters(self):
        prompts = build_introspection_prompt_bank()[:1]
        first = generate_introspection_rollouts(
            FakeIntrospectionRunner(ModelCondition.TARGET_IA, "adapter-a"), prompts
        )
        second = generate_introspection_rollouts(
            FakeIntrospectionRunner(ModelCondition.TARGET_IA, "adapter-b"), prompts
        )

        self.assertTrue(
            {item.rollout_id for item in first}.isdisjoint(
                item.rollout_id for item in second
            )
        )

    def test_semantic_grading_is_condition_blind(self):
        runner = FakeJudgeRunner()
        item = rollout("ria", ModelCondition.TARGET_IA, adapter_name="adapter-a")
        grade = grade_semantic_match(
            runner,
            frozen_label(),
            item,
            config=SemanticGraderConfig(seed_start=7),
        )
        self.assertTrue(grade.semantic_match)
        serialized_prompt = str(runner.messages)
        self.assertNotIn("TARGET_IA", serialized_prompt)
        self.assertNotIn("adapter-a", serialized_prompt)

    def test_semantic_grading_rejects_invented_evidence_quote(self):
        runner = FakeJudgeRunner(evidence_quote="text absent from the response")
        item = rollout("ria", ModelCondition.TARGET_IA, adapter_name="adapter-a")

        with self.assertRaisesRegex(ValueError, "verbatim response span"):
            grade_semantic_match(runner, frozen_label(), item)

    def test_recall_false_positive_unsupported_and_specificity_metrics(self):
        labels = (frozen_label("L1", "adapter-a"), frozen_label("L2", "adapter-b"))
        rollouts = (
            rollout("a-ia", ModelCondition.TARGET_IA, adapter_name="adapter-a"),
            rollout("b-ia", ModelCondition.TARGET_IA, adapter_name="adapter-b"),
            rollout("base-ia", ModelCondition.BASE_IA, adapter_name=None),
            rollout("a-self", ModelCondition.TARGET, adapter_name="adapter-a"),
        )
        grades = (
            semantic_grade("L1", rollouts[0], match=True, broad=True, unsupported=("Claims alien values",)),
            semantic_grade("L2", rollouts[0], match=True, unsupported=("Claims alien values",)),
            semantic_grade("L1", rollouts[1], match=False),
            semantic_grade("L2", rollouts[1], match=True, broad=True, narrow=False),
            semantic_grade("L1", rollouts[2], match=True),
            semantic_grade("L2", rollouts[2], match=False),
            semantic_grade("L1", rollouts[3], match=False),
            semantic_grade("L2", rollouts[3], match=False),
        )
        metrics = compute_meta_ia_metrics(
            grades,
            rollouts,
            labels,
            recall_ks=(1,),
        )
        self.assertEqual(metrics.verified_label_recall, 1.0)
        self.assertEqual(metrics.recall_at_k[1], 1.0)
        self.assertEqual(metrics.base_false_positive_rate, 1.0)
        self.assertAlmostEqual(metrics.unsupported_prediction_rate, 0.6)
        self.assertAlmostEqual(metrics.adapter_specificity, 2 / 3)
        self.assertEqual(metrics.cross_adapter_false_positive_rate, 0.5)
        self.assertEqual(metrics.broad_behavior_report_rate, 1.0)
        self.assertEqual(metrics.target_self_report_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
