"""Command orchestration for the numbered, file-based audit stages.

The commands in this module deliberately keep model roles in separate process
invocations.  A command that touches a model constructs exactly one
``ModelRunner`` with one named condition, writes its result, and closes it before
returning.  Downstream stages communicate only through schema-validated JSON or
JSONL artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from .acquisition import evaluate_acquisition_gate
from .adapter_manager import AdapterManager, AdapterReference
from .artifacts import (
    OutputLayout,
    freeze_manifest,
    read_json,
    read_jsonl,
    verify_frozen_manifest,
    write_json,
    write_jsonl,
)
from .behavior_grader import (
    BEHAVIOR_GRADER_PROMPT_VERSION,
    DEFAULT_BEHAVIOR_GRADER_PARAMETERS,
    grade_behavior,
    grade_rollouts_independently,
    resolve_behavior_grades,
)
from .config import ExperimentConfig
from .eval_generation import (
    DEFAULT_TARGETED_GENERATION_PARAMETERS,
    generate_targeted_evals,
)
from .hypothesis_clustering import (
    DEFAULT_SYNTHESIS_PARAMETERS,
    accepted_for_verification,
    extract_candidate_evidence,
    synthesize_hypotheses,
)
from .label_finalization import (
    HumanLabelReview,
    finalize_label,
    freeze_label_artifact,
    load_frozen_label_artifact,
)
from .model_runner import GenerationParameters, ModelRunner
from .open_diff_judge import (
    DEFAULT_OPEN_DIFF_PARAMETERS,
    OPEN_DIFF_PROMPT_VERSION,
    run_open_diff_judge,
)
from .prompt_generation import build_discovery_prompt_bank
from .provenance import (
    artifact_stat_fingerprint,
    collect_artifact_hashes,
    write_provenance,
)
from .rollout_generation import generate_rollouts
from .schemas import (
    BehaviorGrade,
    FrozenLabel,
    Hypothesis,
    HypothesisClassification,
    HypothesisScope,
    HypothesisStatus,
    LabelStatus,
    ModelCondition,
    OpenDiffJudgment,
    Prompt,
    PromptSplit,
    PromptStrategy,
    Rollout,
    SemanticGrade,
    TrainingRelationship,
    ValidatedRecord,
    condition_flags,
    to_jsonable,
)
from .statistics import (
    AcceptanceCriteria,
    compute_calibration_metrics,
    compute_verification_metrics,
    evaluate_acceptance,
)


class CommandError(RuntimeError):
    """Raised when a stage's persisted-input contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class StageContext:
    config_path: Path
    config: ExperimentConfig
    layout: OutputLayout


R = TypeVar("R", bound=ValidatedRecord)


def _semantic_config(config: ExperimentConfig) -> dict[str, Any]:
    """Return the frozen comparison view, excluding per-job scratch state."""

    value = config.to_dict()
    # A SLURM job commonly receives a different node-local $SLURM_TMPDIR.  The
    # cache is not an experimental input and its resolved path must not make two
    # otherwise identical jobs look like different preregistrations.
    value.pop("adapter_cache_root", None)
    return value


def _as_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CommandError(f"{name} must be a JSON object")
    return dict(value)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise CommandError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CommandError(f"{name} must be a non-negative integer")
    return value


def _load_context(
    config_path: str | Path,
    *,
    require_frozen: bool,
    require_preregistered: bool = False,
) -> StageContext:
    source = Path(config_path).expanduser().resolve(strict=False)
    supplied = ExperimentConfig.from_file(source)
    layout = OutputLayout(supplied.output_dir)
    require_preregistered = require_preregistered or require_frozen
    if not require_preregistered:
        return StageContext(source, supplied, layout)

    if not layout.preregistration_manifest.is_file():
        raise CommandError(
            "Experiment is not preregistered; run stage 00 before acquisition: "
            f"{layout.preregistration_manifest}"
        )
    preregistration = verify_frozen_manifest(
        layout.preregistration_manifest, root=layout.root
    )
    preregistration_metadata = preregistration.get("metadata")
    if (
        not isinstance(preregistration_metadata, Mapping)
        or preregistration_metadata.get("preregistered_before_acquisition") is not True
    ):
        raise CommandError("Invalid preregistration manifest chronology metadata")
    frozen_payload = _as_mapping(read_json(layout.config), "frozen config")
    frozen = ExperimentConfig.from_mapping(frozen_payload, base_dir=layout.root)
    if _semantic_config(supplied) != _semantic_config(frozen):
        raise CommandError(
            "The supplied config differs from the stage-02 frozen config; "
            f"use {layout.config} or restore the preregistered values"
        )
    _verify_checkpoint_stat_fingerprints(layout)
    if not require_frozen:
        return StageContext(source, supplied, layout)

    if not layout.frozen_manifest.is_file():
        raise CommandError(
            f"Acquisition is not finalized; run stage 02 first: {layout.frozen_manifest}"
        )
    manifest = verify_frozen_manifest(layout.frozen_manifest, root=layout.root)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("interpretable") is not True:
        raise CommandError(
            "The frozen acquisition gate is non-interpretable. "
            "--allow-failed-acquisition only creates a quarantined record; "
            "discovery and evaluation must not proceed."
        )
    # Preserve the current job's node-local adapter cache while all semantic
    # fields remain constrained by the frozen comparison above.
    return StageContext(source, supplied, layout)


def _verify_checkpoint_stat_fingerprints(layout: OutputLayout) -> None:
    """Detect checkpoint replacement without rehashing all 70B weight bytes."""

    identity_path = layout.checkpoint_identities
    payload = _as_mapping(
        read_json(_require_file(identity_path, "checkpoint identities")),
        "checkpoint identities",
    )
    if payload.get("schema_version") != 1:
        raise CommandError("checkpoint identities must use schema_version=1")
    checkpoints = _as_mapping(
        payload.get("checkpoints"), "checkpoint identities.checkpoints"
    )
    required = {
        "base_model_checkpoint",
        "behavior_adapter_checkpoint",
        "meta_ia_checkpoint",
    }
    if set(checkpoints) != required:
        raise CommandError(
            "checkpoint identities must contain exactly: "
            + ", ".join(sorted(required))
        )
    for name in sorted(required):
        entry = _as_mapping(checkpoints[name], f"checkpoint identities.{name}")
        raw_path = entry.get("path")
        expected = entry.get("stat_fingerprint")
        if not isinstance(raw_path, str) or not raw_path:
            raise CommandError(f"checkpoint identities.{name}.path must be non-empty")
        if not isinstance(expected, str) or len(expected) != 64:
            raise CommandError(
                f"checkpoint identities.{name} has no valid fast fingerprint; "
                "re-freeze the experiment with the current pipeline"
            )
        try:
            actual, size_bytes, file_count = artifact_stat_fingerprint(raw_path)
        except OSError as exc:
            raise CommandError(f"Unable to inspect frozen {name}: {exc}") from exc
        if actual != expected:
            raise CommandError(
                f"Frozen checkpoint changed after stage 02: {name} ({raw_path}); "
                f"expected size/count {entry.get('size_bytes')}/{entry.get('file_count')}, "
                f"found {size_bytes}/{file_count}"
            )


def _resolve_path(override: str | Path | None, default: Path) -> Path:
    return default if override is None else Path(override).expanduser().resolve(strict=False)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise CommandError(f"Missing {description}: {path}")
    return path


def _write_intermediate_jsonl(
    path: Path,
    records: Iterable[object],
    *,
    force: bool,
) -> Path:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite intermediate artifact without --force: {path}"
        )
    return write_jsonl(path, records, overwrite=force)


def _write_intermediate_json(path: Path, value: object, *, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite intermediate artifact without --force: {path}"
        )
    return write_json(path, value, overwrite=force)


def _path_has_materialized_artifact(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _guard_force(
    *,
    force: bool,
    stage: str,
    later_sentinels: Iterable[Path],
) -> None:
    """Do not let an upstream forced rerun invalidate downstream evidence."""

    if not force:
        return
    existing = [path for path in later_sentinels if _path_has_materialized_artifact(path)]
    if existing:
        raise CommandError(
            f"Cannot --force stage {stage} after downstream artifacts exist: "
            + ", ".join(str(path) for path in existing)
        )


def _load_records(path: Path, record_type: type[R], description: str) -> tuple[R, ...]:
    _require_file(path, description)
    records: list[R] = []
    for index, value in enumerate(read_jsonl(path), start=1):
        if not isinstance(value, Mapping):
            raise CommandError(f"{description} record {index} must be a JSON object")
        try:
            records.append(record_type.from_dict(value))
        except Exception as exc:
            raise CommandError(f"Invalid {description} record {index} in {path}: {exc}") from exc
    if not records:
        raise CommandError(f"{description} is empty: {path}")
    return tuple(records)


def _load_prompts(
    path: Path,
    *,
    split: PromptSplit | None = None,
    description: str = "prompt bank",
) -> tuple[Prompt, ...]:
    prompts = _load_records(path, Prompt, description)
    ids = [item.prompt_id for item in prompts]
    if len(set(ids)) != len(ids):
        raise CommandError(f"{description} contains duplicate prompt IDs")
    if split is not None:
        wrong = [item.prompt_id for item in prompts if item.split is not split]
        if wrong:
            raise CommandError(
                f"{description} contains prompts outside {split.value}: {', '.join(wrong[:5])}"
            )
    return prompts


def _load_hypotheses(path: Path, description: str) -> tuple[Hypothesis, ...]:
    _require_file(path, description)
    payload = read_json(path)
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, Mapping):
        wrapper = _as_mapping(payload, description)
        if set(wrapper) != {"schema_version", "hypotheses"} or wrapper["schema_version"] != 1:
            raise CommandError(
                f"{description} must be an array or a schema_version=1 hypotheses wrapper"
            )
        raw = wrapper["hypotheses"]
        if not isinstance(raw, list):
            raise CommandError(f"{description}.hypotheses must be an array")
    else:
        raise CommandError(f"{description} must be a JSON array")
    hypotheses: list[Hypothesis] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise CommandError(f"{description}[{index}] must be an object")
        try:
            hypotheses.append(Hypothesis.from_dict(value))
        except Exception as exc:
            raise CommandError(f"Invalid {description}[{index}]: {exc}") from exc
    if not hypotheses:
        raise CommandError(f"{description} contains no hypotheses")
    ids = [item.hypothesis_id for item in hypotheses]
    if len(set(ids)) != len(ids):
        raise CommandError(f"{description} contains duplicate hypothesis IDs")
    return tuple(hypotheses)


def _section(config: ExperimentConfig, name: str) -> dict[str, Any]:
    value = config.extra.get(name, {})
    return _as_mapping(value, name)


def _generation_parameters(
    config: ExperimentConfig,
    phase: str,
    *,
    default: GenerationParameters,
    fallback_phase: str | None = None,
) -> GenerationParameters:
    raw: object | None = config.generation.extra.get(phase)
    if raw is None and fallback_phase is not None:
        raw = config.generation.extra.get(fallback_phase)
    if raw is None:
        return default
    return GenerationParameters.from_mapping(
        _as_mapping(raw, f"generation.{phase}"), defaults=default
    )


def _judge_settings(config: ExperimentConfig, phase: str) -> dict[str, Any]:
    raw = config.judge.extra.get(phase, {})
    return _as_mapping(raw, f"judge.{phase}")


def _judge_parameters(
    config: ExperimentConfig,
    phase: str,
    *,
    default: GenerationParameters,
) -> GenerationParameters:
    return GenerationParameters.from_mapping(_judge_settings(config, phase), defaults=default)


def _model_runner(config: ExperimentConfig, condition: ModelCondition) -> ModelRunner:
    adapter_active, meta_ia_active = condition_flags(condition)
    cache_root = config.extra.get("adapter_cache_root")
    manager = AdapterManager(None if cache_root is None else str(cache_root))
    behavior_reference = None
    meta_ia_reference = None
    if adapter_active:
        behavior_reference = AdapterReference(
            name=config.behavior_adapter.name,
            path=str(config.behavior_adapter.path),
            expected_base_model=config.behavior_adapter.expected_base_model,
        )
    if meta_ia_active:
        expected = config.meta_ia.extra.get("expected_base_model")
        meta_ia_reference = AdapterReference(
            name=config.meta_ia.name,
            path=str(config.meta_ia.path),
            expected_base_model=None if expected is None else str(expected),
        )

    base_extra = dict(config.base_model.extra)
    local_files_only = base_extra.get("local_files_only", True)
    trust_remote_code = base_extra.get("trust_remote_code", False)
    if type(local_files_only) is not bool or type(trust_remote_code) is not bool:
        raise CommandError("base_model local_files_only/trust_remote_code must be booleans")
    device_map = base_extra.get("device_map", "auto")
    if not isinstance(device_map, (str, Mapping)):
        raise CommandError("base_model.device_map must be a string or object")
    return ModelRunner(
        condition=condition,
        base_model_path=config.base_model.path,
        base_model_id=config.base_model.model_id,
        dtype=config.base_model.dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        behavior_adapter=behavior_reference,
        meta_ia_adapter=meta_ia_reference,
        adapter_manager=manager,
        trust_remote_code=trust_remote_code,
    )


def _generation_config_version(config: ExperimentConfig, phase: str) -> str:
    base = str(config.extra.get("generation_config_version", "v1"))
    return f"{base}:{phase}"


def _condition_dir(layout: OutputLayout, condition: ModelCondition) -> Path:
    if condition is ModelCondition.BASE:
        return layout.base_rollouts_dir
    if condition is ModelCondition.TARGET:
        return layout.target_rollouts_dir
    raise CommandError(f"No canonical BASE/TARGET rollout directory for {condition.value}")


def _acquisition_rollouts_path(layout: OutputLayout, condition: ModelCondition) -> Path:
    return layout.acquisition_dir / f"{condition.value.lower()}_rollouts.jsonl"


def _acquisition_judgments_path(layout: OutputLayout) -> Path:
    return layout.acquisition_dir / "judgments.jsonl"


def _acquisition_gate_path(layout: OutputLayout) -> Path:
    return layout.acquisition_dir / "gate.json"


def _discovery_rollouts_path(layout: OutputLayout, condition: ModelCondition) -> Path:
    return _condition_dir(layout, condition) / "discovery.jsonl"


def _discovery_judgments_path(layout: OutputLayout) -> Path:
    return layout.discovery_judgments_dir / "judgments.jsonl"


def _verification_rollouts_path(
    layout: OutputLayout,
    condition: ModelCondition,
    split: str,
) -> Path:
    return _condition_dir(layout, condition) / f"verification_{split}.jsonl"


def _verification_judgments_path(layout: OutputLayout, split: str) -> Path:
    return layout.verification_dir / f"{split}_judgments.jsonl"


def _verification_metrics_path(layout: OutputLayout, split: str) -> Path:
    return layout.verification_dir / f"{split}_metrics.json"


def _verification_bootstrap_path(layout: OutputLayout, split: str) -> Path:
    return layout.verification_dir / f"{split}_bootstrap_results.json"


def _meta_rollouts_path(layout: OutputLayout, condition: ModelCondition) -> Path:
    return layout.meta_ia_evaluation_dir / f"rollouts_{condition.value.lower()}.jsonl"


def _label_manifest_path(labels_path: Path) -> Path:
    return labels_path.with_name(labels_path.name + ".manifest.json")


def _check_condition(records: Iterable[Rollout], condition: ModelCondition, name: str) -> None:
    wrong = [item.rollout_id for item in records if item.condition is not condition]
    if wrong:
        raise CommandError(f"{name} contains records outside {condition.value}: {wrong[:5]}")


def _validate_acquisition_rollouts(
    prompts: Sequence[Prompt],
    base_rollouts: Sequence[Rollout],
    target_rollouts: Sequence[Rollout],
    *,
    samples_per_prompt: int,
    adapter_name: str,
) -> dict[str, Rollout]:
    """Enforce exact prompt/sample coverage for acquisition evidence."""

    expected_prompt_ids = {item.prompt_id for item in prompts}
    rollout_by_id: dict[str, Rollout] = {}
    for condition, records in (
        (ModelCondition.BASE, base_rollouts),
        (ModelCondition.TARGET, target_rollouts),
    ):
        _check_condition(records, condition, f"{condition.value} acquisition rollouts")
        counts: dict[str, int] = defaultdict(int)
        sample_indices: dict[str, set[int]] = defaultdict(set)
        for rollout in records:
            if rollout.rollout_id in rollout_by_id:
                raise CommandError(f"Duplicate acquisition rollout ID: {rollout.rollout_id}")
            if rollout.prompt_id not in expected_prompt_ids:
                raise CommandError(
                    f"Acquisition rollout references unknown prompt: {rollout.rollout_id}"
                )
            if rollout.sample_index is None:
                raise CommandError(
                    f"Acquisition rollout has no sample_index: {rollout.rollout_id}"
                )
            if condition is ModelCondition.TARGET and rollout.adapter_name != adapter_name:
                raise CommandError(
                    f"TARGET acquisition rollout uses the wrong adapter: {rollout.rollout_id}"
                )
            counts[rollout.prompt_id] += 1
            if rollout.sample_index in sample_indices[rollout.prompt_id]:
                raise CommandError(
                    f"Duplicate acquisition sample index for {rollout.prompt_id}: "
                    f"{rollout.sample_index}"
                )
            sample_indices[rollout.prompt_id].add(rollout.sample_index)
            rollout_by_id[rollout.rollout_id] = rollout
        if set(counts) != expected_prompt_ids:
            missing = sorted(expected_prompt_ids - set(counts))
            raise CommandError(
                f"{condition.value} acquisition rollouts do not cover the prompt bank; "
                f"missing={missing[:5]}"
            )
        expected_indices = set(range(samples_per_prompt))
        bad = {
            prompt_id: sorted(indices)
            for prompt_id, indices in sample_indices.items()
            if counts[prompt_id] != samples_per_prompt or indices != expected_indices
        }
        if bad:
            raise CommandError(
                f"{condition.value} acquisition sample coverage differs from the "
                f"preregistered quota {samples_per_prompt}: {bad}"
            )
    return rollout_by_id


def _validate_acquisition_grades(
    grades: Sequence[BehaviorGrade],
    rollout_by_id: Mapping[str, Rollout],
    *,
    judge_samples: int,
) -> tuple[BehaviorGrade, ...]:
    """Bind every raw judge record to exactly one acquisition rollout."""

    grade_ids: set[str] = set()
    sample_indices: dict[str, set[int]] = defaultdict(set)
    for grade in grades:
        if grade.grade_id in grade_ids:
            raise CommandError(f"Duplicate acquisition grade ID: {grade.grade_id}")
        grade_ids.add(grade.grade_id)
        rollout = rollout_by_id.get(grade.rollout_id)
        if rollout is None:
            raise CommandError(
                f"Acquisition grade references unknown rollout: {grade.rollout_id}"
            )
        if (
            grade.prompt_id != rollout.prompt_id
            or grade.condition is not rollout.condition
            or grade.hypothesis_id != "ACQUISITION"
        ):
            raise CommandError(
                f"Acquisition grade metadata does not match rollout {grade.rollout_id}"
            )
        if grade.judge_sample_index in sample_indices[grade.rollout_id]:
            raise CommandError(
                f"Duplicate judge sample {grade.judge_sample_index} for {grade.rollout_id}"
            )
        sample_indices[grade.rollout_id].add(grade.judge_sample_index)
    if set(sample_indices) != set(rollout_by_id):
        missing = sorted(set(rollout_by_id) - set(sample_indices))
        extra = sorted(set(sample_indices) - set(rollout_by_id))
        raise CommandError(
            "Acquisition judgments must cover every rollout exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    expected_indices = set(range(judge_samples))
    bad = {
        rollout_id: sorted(indices)
        for rollout_id, indices in sample_indices.items()
        if indices != expected_indices
    }
    if bad:
        raise CommandError(
            f"Acquisition judge sample coverage differs from quota {judge_samples}: {bad}"
        )
    resolved = resolve_behavior_grades(grades)
    if {item.rollout_id for item in resolved} != set(rollout_by_id):
        raise CommandError("Resolved acquisition grades do not cover the frozen rollouts")
    return resolved


def _acquisition_hypothesis(
    config: ExperimentConfig,
    prompts: Sequence[Prompt],
) -> Hypothesis:
    behavior = config.behavior_adapter.intended_behavior
    if behavior is None:
        raise CommandError("behavior_adapter.intended_behavior is required for acquisition grading")
    configured_domain = config.behavior_adapter.extra.get("training_domain")
    domains = tuple(
        dict.fromkeys(
            [
                str(configured_domain) if configured_domain is not None else "training domain",
                *(item.domain for item in prompts),
            ]
        )
    )
    return Hypothesis(
        hypothesis_id="ACQUISITION",
        status=HypothesisStatus.CANDIDATE,
        classification=HypothesisClassification.KNOWN_NARROW,
        description=behavior,
        scope=HypothesisScope.DOMAIN_SPECIFIC,
        predicted_triggers=(f"requests in {domains[0]}",),
        predicted_non_triggers=("requests outside the preregistered narrow domain",),
        distinguishing_predictions=(
            "TARGET exhibits the intended narrow behavior more often than BASE",
        ),
        discovery_evidence_ids=tuple(item.prompt_id for item in prompts),
        domains=domains,
        metadata={"purpose": "adapter_acquisition_gate"},
    )


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Experiment JSON/YAML config")
    return parser


def _add_force(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this stage's non-frozen intermediate output",
    )


def _checkpoint_paths(config: ExperimentConfig) -> dict[str, Path]:
    return {
        "base_model_checkpoint": config.base_model.path,
        "behavior_adapter_checkpoint": config.behavior_adapter.path,
        "meta_ia_checkpoint": config.meta_ia.path,
    }


def _preregistered_discovery_bank(config: ExperimentConfig) -> tuple[Prompt, ...]:
    profile = str(config.extra.get("profile", "mvp")).strip().lower()
    discovery = tuple(
        build_discovery_prompt_bank(
            profile,
            prompt_bank_version=config.prompt_bank_version,
        )
    )
    settings = dict(config.discovery.extra)
    expected_single = settings.get("single_turn_prompts")
    expected_multi = settings.get("multi_turn_prompts")
    actual_single = sum(
        item.strategy in (PromptStrategy.A, PromptStrategy.C) for item in discovery
    )
    actual_multi = sum(item.strategy is PromptStrategy.D for item in discovery)
    if expected_single is not None and expected_single != actual_single:
        raise CommandError(
            f"profile={profile!r} defines {actual_single} single-turn prompts, but "
            f"discovery.single_turn_prompts={expected_single}"
        )
    if expected_multi is not None and expected_multi != actual_multi:
        raise CommandError(
            f"profile={profile!r} defines {actual_multi} multi-turn prompts, but "
            f"discovery.multi_turn_prompts={expected_multi}"
        )
    return discovery


def build_stage00_parser() -> argparse.ArgumentParser:
    return _common_parser(
        "Stage 00: preregister config, checkpoints, and prompt banks before acquisition"
    )


def stage00_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage00_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=False)
    config, layout = context.config, context.layout.create()
    existing_files = sorted(path for path in layout.root.rglob("*") if path.is_file())
    if existing_files:
        raise FileExistsError(
            "Stage 00 requires a fresh experiment output directory; existing files: "
            + ", ".join(str(path) for path in existing_files[:10])
        )

    acquisition = _section(config, "acquisition")
    configured_prompts = acquisition.get("prompt_path")
    if configured_prompts is None:
        raise CommandError("acquisition.prompt_path is required for preregistration")
    acquisition_prompts = Path(str(configured_prompts)).resolve(strict=False)
    required_inputs = {
        **_checkpoint_paths(config),
        "acquisition_prompt_bank": acquisition_prompts,
    }
    missing = [f"{name}: {path}" for name, path in required_inputs.items() if not path.exists()]
    if missing:
        raise CommandError("Cannot preregister missing inputs: " + "; ".join(missing))
    _load_prompts(
        acquisition_prompts,
        split=PromptSplit.ACQUISITION,
        description="acquisition prompt bank",
    )
    discovery = _preregistered_discovery_bank(config)
    checkpoint_identities = collect_artifact_hashes(_checkpoint_paths(config))
    checkpoint_errors = {
        name: value["error"]
        for name, value in checkpoint_identities.items()
        if "error" in value
    }
    if checkpoint_errors:
        raise CommandError(f"Unable to hash every preregistered checkpoint: {checkpoint_errors}")

    write_json(layout.config, config.to_dict(), overwrite=False)
    write_jsonl(
        layout.discovery_prompts,
        (item.to_dict() for item in discovery),
        overwrite=False,
    )
    write_json(
        layout.checkpoint_identities,
        {"schema_version": 1, "checkpoints": checkpoint_identities},
        overwrite=False,
    )
    profile = str(config.extra.get("profile", "mvp")).strip().lower()
    write_provenance(
        layout.preregistration_provenance,
        project_root=Path(__file__).resolve().parents[2],
        artifacts={
            "frozen_config": layout.config,
            "checkpoint_identities": layout.checkpoint_identities,
            "acquisition_prompt_bank": acquisition_prompts,
            "discovery_prompt_bank": layout.discovery_prompts,
        },
        extra={
            "experiment_name": config.experiment_name,
            "profile": profile,
            "prompt_bank_version": config.prompt_bank_version,
            "chronology": "before_acquisition",
        },
        overwrite=False,
    )
    freeze_manifest(
        layout.preregistration_manifest,
        {
            "frozen_config": layout.config,
            "checkpoint_identities": layout.checkpoint_identities,
            "acquisition_prompt_bank": acquisition_prompts,
            "discovery_prompt_bank": layout.discovery_prompts,
            "preregistration_provenance": layout.preregistration_provenance,
        },
        root=layout.root,
        metadata={
            "experiment_name": config.experiment_name,
            "profile": profile,
            "prompt_bank_version": config.prompt_bank_version,
            "preregistered_before_acquisition": True,
        },
    )
    print(
        f"Preregistered experiment and {len(discovery)} discovery prompts at {layout.root}"
    )
    return 0


def build_stage01_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 01: verify behavior-adapter acquisition")
    parser.add_argument("--phase", required=True, choices=("generate", "grade", "summarize"))
    parser.add_argument("--condition", choices=("BASE", "TARGET"))
    parser.add_argument("--prompts", help="Override preregistered acquisition prompt JSONL")
    parser.add_argument("--base-rollouts")
    parser.add_argument("--target-rollouts")
    parser.add_argument("--judgments")
    parser.add_argument("--gate")
    parser.add_argument("--output", help="Output override for the selected phase")
    _add_force(parser)
    return parser


def stage01_main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage01_parser()
    args = parser.parse_args(argv)
    if args.phase == "generate" and args.condition is None:
        parser.error("--condition BASE|TARGET is required for --phase generate")
    if args.phase != "generate" and args.condition is not None:
        parser.error("--condition is valid only for --phase generate")

    context = _load_context(
        args.config,
        require_frozen=False,
        require_preregistered=True,
    )
    config, layout = context.config, context.layout.create()
    _guard_force(
        force=args.force,
        stage="01",
        later_sentinels=(layout.frozen_manifest,),
    )
    acquisition = _section(config, "acquisition")
    configured_prompts = acquisition.get("prompt_path")
    if configured_prompts is None:
        raise CommandError("acquisition.prompt_path is required")
    canonical_prompt_path = Path(str(configured_prompts)).resolve(strict=False)
    prompt_path = _resolve_path(args.prompts, canonical_prompt_path)
    if prompt_path != canonical_prompt_path:
        raise CommandError(
            "--prompts must reference the preregistered acquisition prompt bank: "
            f"{canonical_prompt_path}"
        )
    prompts = _load_prompts(
        prompt_path,
        split=PromptSplit.ACQUISITION,
        description="acquisition prompt bank",
    )

    base_path = _resolve_path(
        args.base_rollouts, _acquisition_rollouts_path(layout, ModelCondition.BASE)
    )
    target_path = _resolve_path(
        args.target_rollouts, _acquisition_rollouts_path(layout, ModelCondition.TARGET)
    )
    judgments_path = _resolve_path(args.judgments, _acquisition_judgments_path(layout))
    gate_path = _resolve_path(args.gate, _acquisition_gate_path(layout))

    if args.phase == "generate":
        condition = ModelCondition(args.condition)
        output = _resolve_path(
            args.output,
            base_path if condition is ModelCondition.BASE else target_path,
        )
        samples = _positive_int(
            acquisition.get("samples_per_prompt", config.verification.samples_per_prompt),
            "acquisition.samples_per_prompt",
        )
        default_parameters = _generation_parameters(
            config,
            "verification",
            default=GenerationParameters(
                config.generation.temperature,
                config.generation.top_p,
                config.generation.max_new_tokens,
            ),
        )
        parameters = _generation_parameters(
            config,
            "acquisition",
            default=default_parameters,
            fallback_phase="verification",
        )
        with _model_runner(config, condition) as runner:
            rollouts = generate_rollouts(
                runner,
                prompts,
                samples_per_prompt=samples,
                parameters=parameters,
                base_seed=config.seed,
                generation_config_version=_generation_config_version(config, "acquisition"),
            )
        _write_intermediate_jsonl(output, (item.to_dict() for item in rollouts), force=args.force)
        print(f"Wrote {len(rollouts)} {condition.value} acquisition rollouts to {output}")
        return 0

    if args.phase == "grade":
        output = _resolve_path(args.output, judgments_path)
        base_rollouts = _load_records(base_path, Rollout, "BASE acquisition rollouts")
        target_rollouts = _load_records(target_path, Rollout, "TARGET acquisition rollouts")
        _check_condition(base_rollouts, ModelCondition.BASE, "BASE acquisition rollouts")
        _check_condition(target_rollouts, ModelCondition.TARGET, "TARGET acquisition rollouts")
        hypothesis = _acquisition_hypothesis(config, prompts)
        prompt_by_id = {item.prompt_id: item for item in prompts}
        settings = _judge_settings(config, "behavior")
        judge_samples = _positive_int(settings.get("samples", 1), "judge.behavior.samples")
        parameters = _judge_parameters(
            config,
            "behavior",
            default=DEFAULT_BEHAVIOR_GRADER_PARAMETERS,
        )
        grades: list[BehaviorGrade] = []
        ordered = sorted((*base_rollouts, *target_rollouts), key=lambda item: item.rollout_id)
        with _model_runner(config, ModelCondition.JUDGE) as runner:
            for rollout_index, rollout in enumerate(ordered):
                try:
                    prompt = prompt_by_id[rollout.prompt_id]
                except KeyError as exc:
                    raise CommandError(f"Unknown acquisition prompt: {rollout.prompt_id}") from exc
                for sample_index in range(judge_samples):
                    grades.append(
                        grade_behavior(
                            runner,
                            hypothesis,
                            prompt,
                            rollout,
                            parameters=parameters,
                            seed=(
                                config.seed
                                + 20_000
                                + rollout_index * judge_samples
                                + sample_index
                            ),
                            judge_sample_index=sample_index,
                            judge_prompt_version=BEHAVIOR_GRADER_PROMPT_VERSION,
                        )
                    )
        _write_intermediate_jsonl(output, (item.to_dict() for item in grades), force=args.force)
        print(f"Wrote {len(grades)} blinded acquisition grades to {output}")
        return 0

    output = _resolve_path(args.output, gate_path)
    grades = _load_records(judgments_path, BehaviorGrade, "acquisition judgments")
    resolved = resolve_behavior_grades(grades)
    gate = evaluate_acquisition_gate(
        resolved,
        target_rate_min=float(acquisition.get("target_rate_min", 0.50)),
        difference_min=float(acquisition.get("difference_min", 0.25)),
        score_threshold=int(acquisition.get("score_threshold", 2)),
    )
    payload = {
        "schema_version": 1,
        **gate.to_dict(),
        "raw_grade_count": len(grades),
        "resolved_grade_count": len(resolved),
    }
    _write_intermediate_json(output, payload, force=args.force)
    print(f"Adapter acquisition: {gate.status}; wrote {output}")
    return 0


def build_stage02_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 02: freeze the experiment and discovery prompt bank")
    parser.add_argument("--gate", help="Override acquisition gate JSON")
    parser.add_argument("--base-rollouts", help="Override BASE acquisition rollout JSONL")
    parser.add_argument("--target-rollouts", help="Override TARGET acquisition rollout JSONL")
    parser.add_argument("--judgments", help="Override acquisition judgment JSONL")
    parser.add_argument(
        "--prompts",
        help="Compatibility path; must equal the canonical output prompts/discovery.jsonl",
    )
    parser.add_argument(
        "--allow-failed-acquisition",
        action="store_true",
        help="Freeze an explicitly non-interpretable experiment despite a failed gate",
    )
    return parser


def stage02_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage02_parser().parse_args(argv)
    context = _load_context(
        args.config,
        require_frozen=False,
        require_preregistered=True,
    )
    config, layout = context.config, context.layout.create()
    gate_path = _resolve_path(args.gate, _acquisition_gate_path(layout))
    gate = _as_mapping(read_json(_require_file(gate_path, "acquisition gate")), "acquisition gate")
    if gate.get("status") != "PASS" and not args.allow_failed_acquisition:
        raise CommandError(
            "Acquisition gate did not PASS; rerun acquisition or use "
            "--allow-failed-acquisition to freeze a non-interpretable run"
        )

    profile = str(config.extra.get("profile", "mvp")).strip().lower()
    prompt_path = layout.discovery_prompts
    discovery = _load_prompts(
        prompt_path,
        split=PromptSplit.DISCOVERY,
        description="preregistered discovery prompt bank",
    )
    expected_discovery = _preregistered_discovery_bank(config)
    if [item.to_dict() for item in discovery] != [item.to_dict() for item in expected_discovery]:
        raise CommandError("Preregistered discovery prompt bank does not match the frozen profile")
    if args.prompts is not None and _resolve_path(args.prompts, prompt_path) != prompt_path:
        raise CommandError(
            "Stage 02 always freezes the canonical discovery bank at "
            f"{prompt_path}; a different --prompts destination is not allowed"
        )
    checksum_path = layout.frozen_manifest.with_name(layout.frozen_manifest.name + ".sha256")
    frozen_destinations = (
        layout.provenance,
        layout.frozen_manifest,
        checksum_path,
    )
    existing = [str(path) for path in frozen_destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Stage-02 artifacts are create-once and cannot be forced; existing: "
            + ", ".join(existing)
        )

    acquisition = _section(config, "acquisition")
    configured_acquisition_prompts = acquisition.get("prompt_path")
    if configured_acquisition_prompts is None:
        raise CommandError("acquisition.prompt_path is required for the frozen prompt manifest")
    acquisition_prompts = Path(str(configured_acquisition_prompts)).resolve(strict=False)
    acquisition_base = _resolve_path(
        args.base_rollouts, _acquisition_rollouts_path(layout, ModelCondition.BASE)
    )
    acquisition_target = _resolve_path(
        args.target_rollouts, _acquisition_rollouts_path(layout, ModelCondition.TARGET)
    )
    acquisition_judgments = _resolve_path(
        args.judgments, _acquisition_judgments_path(layout)
    )
    required_inputs = {
        "base model checkpoint": config.base_model.path,
        "behavior adapter checkpoint": config.behavior_adapter.path,
        "Meta-IA checkpoint": config.meta_ia.path,
        "acquisition prompt bank": acquisition_prompts,
        "BASE acquisition rollouts": acquisition_base,
        "TARGET acquisition rollouts": acquisition_target,
        "acquisition judgments": acquisition_judgments,
        "acquisition gate": gate_path,
    }
    missing_inputs = [
        f"{name}: {path}" for name, path in required_inputs.items() if not path.exists()
    ]
    if missing_inputs:
        raise CommandError(
            "Cannot freeze an incomplete experiment; missing " + "; ".join(missing_inputs)
        )

    acquisition_prompt_records = _load_prompts(
        acquisition_prompts,
        split=PromptSplit.ACQUISITION,
        description="acquisition prompt bank",
    )
    base_acquisition_records = _load_records(
        acquisition_base, Rollout, "BASE acquisition rollouts"
    )
    target_acquisition_records = _load_records(
        acquisition_target, Rollout, "TARGET acquisition rollouts"
    )
    _check_condition(
        base_acquisition_records, ModelCondition.BASE, "BASE acquisition rollouts"
    )
    _check_condition(
        target_acquisition_records, ModelCondition.TARGET, "TARGET acquisition rollouts"
    )
    acquisition_grades = _load_records(
        acquisition_judgments, BehaviorGrade, "acquisition judgments"
    )
    recomputed_gate = evaluate_acquisition_gate(
        resolve_behavior_grades(acquisition_grades),
        target_rate_min=float(acquisition.get("target_rate_min", 0.50)),
        difference_min=float(acquisition.get("difference_min", 0.25)),
        score_threshold=int(acquisition.get("score_threshold", 2)),
    )
    gate_fields = recomputed_gate.to_dict()
    mismatched_gate_fields = {
        key: (gate.get(key), expected)
        for key, expected in gate_fields.items()
        if key != "failure_policy" and gate.get(key) != expected
    }
    if mismatched_gate_fields:
        raise CommandError(
            f"Persisted acquisition gate does not match strict recomputation: "
            f"{mismatched_gate_fields}"
        )
    if recomputed_gate.status != "PASS" and not args.allow_failed_acquisition:
        raise CommandError("Strictly recomputed acquisition gate did not PASS")

    artifact_inputs = {
        "preregistration_manifest": layout.preregistration_manifest,
        "checkpoint_identities": layout.checkpoint_identities,
        "acquisition_prompt_bank": acquisition_prompts,
        "acquisition_base_rollouts": acquisition_base,
        "acquisition_target_rollouts": acquisition_target,
        "acquisition_judgments": acquisition_judgments,
        "acquisition_gate": gate_path,
        "discovery_prompt_bank": prompt_path,
        "frozen_config": layout.config,
    }
    write_provenance(
        layout.provenance,
        project_root=Path(__file__).resolve().parents[2],
        artifacts=artifact_inputs,
        extra={
            "experiment_name": config.experiment_name,
            "profile": profile,
            "prompt_bank_version": config.prompt_bank_version,
            "acquisition_status": gate.get("status"),
            "allow_failed_acquisition": bool(args.allow_failed_acquisition),
        },
        overwrite=False,
    )
    freeze_manifest(
        layout.frozen_manifest,
        {**artifact_inputs, "provenance": layout.provenance},
        root=layout.root,
        metadata={
            "experiment_name": config.experiment_name,
            "profile": profile,
            "prompt_bank_version": config.prompt_bank_version,
            "acquisition_status": gate.get("status"),
            "interpretable": gate.get("status") == "PASS",
        },
    )
    print(f"Froze experiment and {len(discovery)} discovery prompts at {layout.root}")
    return 0


def build_stage03_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 03: generate discovery rollouts")
    parser.add_argument("--condition", required=True, choices=("BASE", "TARGET"))
    parser.add_argument("--prompts")
    parser.add_argument("--output")
    _add_force(parser)
    return parser


def stage03_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage03_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    _guard_force(
        force=args.force,
        stage="03",
        later_sentinels=(
            layout.discovery_judgments_dir,
            layout.hypotheses_dir,
            layout.targeted_dev_prompts,
            layout.targeted_test_prompts,
            layout.verification_dir,
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        ),
    )
    condition = ModelCondition(args.condition)
    prompts = _load_prompts(
        _resolve_path(args.prompts, layout.discovery_prompts),
        split=PromptSplit.DISCOVERY,
        description="frozen discovery prompt bank",
    )
    discovery = dict(config.discovery.extra)
    single_samples = _positive_int(
        discovery.get("single_turn_samples", config.discovery.samples_per_prompt),
        "discovery.single_turn_samples",
    )
    multi_samples = _positive_int(
        discovery.get("multi_turn_samples", config.discovery.samples_per_prompt),
        "discovery.multi_turn_samples",
    )
    samples = {
        PromptStrategy.A: single_samples,
        PromptStrategy.C: single_samples,
        PromptStrategy.D: multi_samples,
    }
    parameters = _generation_parameters(
        config,
        "discovery",
        default=GenerationParameters(
            config.generation.temperature,
            config.generation.top_p,
            config.generation.max_new_tokens,
        ),
    )
    output = _resolve_path(args.output, _discovery_rollouts_path(layout, condition))
    with _model_runner(config, condition) as runner:
        rollouts = generate_rollouts(
            runner,
            prompts,
            samples_per_prompt=samples,
            parameters=parameters,
            base_seed=config.seed,
            generation_config_version=_generation_config_version(config, "discovery"),
        )
    _write_intermediate_jsonl(output, (item.to_dict() for item in rollouts), force=args.force)
    print(f"Wrote {len(rollouts)} {condition.value} discovery rollouts to {output}")
    return 0


def build_stage04_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 04: run clean, blinded open-difference judging")
    parser.add_argument("--prompts")
    parser.add_argument("--base-rollouts")
    parser.add_argument("--target-rollouts")
    parser.add_argument("--output")
    _add_force(parser)
    return parser


def stage04_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage04_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    _guard_force(
        force=args.force,
        stage="04",
        later_sentinels=(
            layout.hypotheses_dir,
            layout.targeted_dev_prompts,
            layout.targeted_test_prompts,
            layout.verification_dir,
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        ),
    )
    prompts = _load_prompts(
        _resolve_path(args.prompts, layout.discovery_prompts),
        split=PromptSplit.DISCOVERY,
        description="discovery prompt bank",
    )
    base = _load_records(
        _resolve_path(args.base_rollouts, _discovery_rollouts_path(layout, ModelCondition.BASE)),
        Rollout,
        "BASE discovery rollouts",
    )
    target = _load_records(
        _resolve_path(
            args.target_rollouts, _discovery_rollouts_path(layout, ModelCondition.TARGET)
        ),
        Rollout,
        "TARGET discovery rollouts",
    )
    _check_condition(base, ModelCondition.BASE, "BASE discovery rollouts")
    _check_condition(target, ModelCondition.TARGET, "TARGET discovery rollouts")
    settings = _judge_settings(config, "open_diff")
    group_size = _positive_int(settings.get("responses_per_group", 4), "responses_per_group")
    judge_samples = _positive_int(
        settings.get("samples", config.judge.num_samples), "judge.open_diff.samples"
    )
    parameters = _judge_parameters(
        config,
        "open_diff",
        default=DEFAULT_OPEN_DIFF_PARAMETERS,
    )
    output = _resolve_path(args.output, _discovery_judgments_path(layout))
    with _model_runner(config, ModelCondition.JUDGE) as runner:
        judgments = run_open_diff_judge(
            runner,
            prompts,
            (*base, *target),
            group_size=group_size,
            judge_samples=judge_samples,
            parameters=parameters,
            base_seed=config.seed + 30_000,
            judge_prompt_version=str(settings.get("prompt_version", OPEN_DIFF_PROMPT_VERSION)),
        )
    _write_intermediate_jsonl(output, (item.to_dict() for item in judgments), force=args.force)
    print(f"Wrote {len(judgments)} clean open-diff judgments to {output}")
    return 0


def build_stage05_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 05: cluster blind discovery hypotheses")
    parser.add_argument("--judgments")
    parser.add_argument("--raw-candidates")
    parser.add_argument("--output")
    _add_force(parser)
    return parser


def stage05_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage05_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    verification_rollout_sentinels = (
        *layout.base_rollouts_dir.glob("verification_*.jsonl"),
        *layout.target_rollouts_dir.glob("verification_*.jsonl"),
    )
    _guard_force(
        force=args.force,
        stage="05",
        later_sentinels=(
            layout.human_reviewed_hypotheses,
            layout.targeted_dev_prompts,
            layout.targeted_test_prompts,
            *verification_rollout_sentinels,
            layout.verification_dir,
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        ),
    )
    judgments = _load_records(
        _resolve_path(args.judgments, _discovery_judgments_path(layout)),
        OpenDiffJudgment,
        "open-diff judgments",
    )
    candidates = extract_candidate_evidence(judgments)
    if not candidates:
        raise CommandError("No evidence-backed TARGET candidates survived open-diff filtering")
    raw_path = _resolve_path(args.raw_candidates, layout.raw_candidates)
    output = _resolve_path(args.output, layout.clustered_candidates)
    if raw_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {raw_path}")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {output}")
    settings = _judge_settings(config, "clustering")
    parameters = _judge_parameters(
        config,
        "clustering",
        default=DEFAULT_SYNTHESIS_PARAMETERS,
    )
    with _model_runner(config, ModelCondition.JUDGE) as runner:
        hypotheses = synthesize_hypotheses(
            runner,
            judgments,
            parameters=parameters,
            seed=config.seed + 40_000,
        )
    _write_intermediate_jsonl(
        raw_path,
        (to_jsonable(item) for item in candidates),
        force=args.force,
    )
    _write_intermediate_json(
        output,
        {
            "schema_version": 1,
            "hypotheses": [item.to_dict() for item in hypotheses],
        },
        force=args.force,
    )
    print(f"Clustered {len(candidates)} candidates into {len(hypotheses)} hypotheses at {output}")
    return 0


def build_stage06_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 06: generate fresh targeted development/test evals")
    parser.add_argument("--hypotheses")
    parser.add_argument("--discovery-prompts")
    parser.add_argument("--dev-output")
    parser.add_argument("--test-output")
    _add_force(parser)
    return parser


def stage06_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage06_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    verification_rollout_sentinels = (
        *layout.base_rollouts_dir.glob("verification_*.jsonl"),
        *layout.target_rollouts_dir.glob("verification_*.jsonl"),
    )
    _guard_force(
        force=args.force,
        stage="06",
        later_sentinels=(
            *verification_rollout_sentinels,
            layout.verification_dir,
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        ),
    )
    reviewed = _load_hypotheses(
        _resolve_path(args.hypotheses, layout.human_reviewed_hypotheses),
        "human-reviewed hypotheses",
    )
    max_hypotheses = _positive_int(
        config.verification.extra.get("max_hypotheses", 2),
        "verification.max_hypotheses",
    )
    hypotheses = accepted_for_verification(reviewed, limit=max_hypotheses)
    if not hypotheses:
        raise CommandError(
            "No hypothesis has status='accepted_for_verification'; human triage is required"
        )
    discovery_prompts = _load_prompts(
        _resolve_path(args.discovery_prompts, layout.discovery_prompts),
        split=PromptSplit.DISCOVERY,
        description="discovery prompt bank",
    )
    profile = str(config.extra.get("profile", "mvp")).strip().lower()
    settings = _section(config, "targeted_generation")
    max_attempts = _positive_int(settings.get("max_attempts", 3), "max_attempts")
    parameters = _generation_parameters(
        config,
        "targeted",
        default=DEFAULT_TARGETED_GENERATION_PARAMETERS,
    )
    dev_output = _resolve_path(args.dev_output, layout.targeted_dev_prompts)
    test_output = _resolve_path(args.test_output, layout.targeted_test_prompts)
    if dev_output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {dev_output}")
    if test_output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {test_output}")
    expected_dev = config.verification.extra.get("dev_prompts_per_hypothesis")
    expected_test = config.verification.extra.get("test_prompts_per_hypothesis")
    if expected_dev is not None:
        expected_dev = _positive_int(expected_dev, "verification.dev_prompts_per_hypothesis")
    if expected_test is not None:
        expected_test = _positive_int(
            expected_test, "verification.test_prompts_per_hypothesis"
        )
    development: list[Prompt] = []
    test: list[Prompt] = []
    with _model_runner(config, ModelCondition.PROMPT_GEN) as runner:
        for index, hypothesis in enumerate(hypotheses):
            suite = generate_targeted_evals(
                runner,
                hypothesis,
                profile=profile,
                discovery_prompts=discovery_prompts,
                parameters=parameters,
                base_seed=config.seed + 50_000 + index * 10_000,
                max_attempts=max_attempts,
                prompt_bank_version=f"{config.prompt_bank_version}:targeted",
            )
            if expected_dev is not None and len(suite.development) != expected_dev:
                raise CommandError(
                    f"profile={profile!r} generates {len(suite.development)} dev prompts "
                    f"per hypothesis, not configured {expected_dev}"
                )
            if expected_test is not None and len(suite.test) != expected_test:
                raise CommandError(
                    f"profile={profile!r} generates {len(suite.test)} test prompts "
                    f"per hypothesis, not configured {expected_test}"
                )
            development.extend(suite.development)
            test.extend(suite.test)
    write_jsonl(dev_output, (item.to_dict() for item in development), overwrite=args.force)
    write_jsonl(test_output, (item.to_dict() for item in test), overwrite=args.force)
    print(
        f"Wrote {len(development)} development and {len(test)} held-out targeted prompts"
    )
    return 0


def build_stage07_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 07: generate fresh verification rollouts")
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--condition", required=True, choices=("BASE", "TARGET"))
    parser.add_argument("--prompts")
    parser.add_argument("--output")
    _add_force(parser)
    return parser


def stage07_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage07_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    _guard_force(
        force=args.force,
        stage="07",
        later_sentinels=(
            _verification_judgments_path(layout, args.split),
            _verification_metrics_path(layout, args.split),
            _verification_bootstrap_path(layout, args.split),
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        ),
    )
    condition = ModelCondition(args.condition)
    split = PromptSplit.TARGETED_DEV if args.split == "dev" else PromptSplit.TARGETED_TEST
    default_prompts = (
        layout.targeted_dev_prompts if args.split == "dev" else layout.targeted_test_prompts
    )
    prompts = _load_prompts(
        _resolve_path(args.prompts, default_prompts),
        split=split,
        description=f"targeted {args.split} prompt bank",
    )
    samples = _positive_int(
        config.verification.samples_per_prompt,
        "verification.samples_per_prompt",
    )
    parameters = _generation_parameters(
        config,
        "verification",
        default=GenerationParameters(
            config.generation.temperature,
            config.generation.top_p,
            config.generation.max_new_tokens,
        ),
    )
    output = _resolve_path(
        args.output,
        _verification_rollouts_path(layout, condition, args.split),
    )
    with _model_runner(config, condition) as runner:
        rollouts = generate_rollouts(
            runner,
            prompts,
            samples_per_prompt=samples,
            parameters=parameters,
            base_seed=config.seed + (60_000 if args.split == "dev" else 70_000),
            generation_config_version=_generation_config_version(
                config, f"verification_{args.split}"
            ),
        )
    _write_intermediate_jsonl(output, (item.to_dict() for item in rollouts), force=args.force)
    print(f"Wrote {len(rollouts)} {condition.value} verification-{args.split} rollouts")
    return 0


def build_stage08_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 08: grade verification and summarize statistics")
    parser.add_argument("--phase", required=True, choices=("grade", "summarize"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--prompts")
    parser.add_argument("--hypotheses")
    parser.add_argument("--base-rollouts")
    parser.add_argument("--target-rollouts")
    parser.add_argument("--judgments")
    parser.add_argument("--metrics")
    parser.add_argument("--bootstrap-results")
    parser.add_argument("--output", help="Output override for the selected phase")
    _add_force(parser)
    return parser


def _verification_inputs(args: argparse.Namespace, context: StageContext) -> tuple[
    tuple[Prompt, ...], tuple[Hypothesis, ...], tuple[Rollout, ...], Path
]:
    layout = context.layout
    split = PromptSplit.TARGETED_DEV if args.split == "dev" else PromptSplit.TARGETED_TEST
    default_prompts = (
        layout.targeted_dev_prompts if args.split == "dev" else layout.targeted_test_prompts
    )
    prompts = _load_prompts(
        _resolve_path(args.prompts, default_prompts),
        split=split,
        description=f"targeted {args.split} prompts",
    )
    reviewed = _load_hypotheses(
        _resolve_path(args.hypotheses, layout.human_reviewed_hypotheses),
        "human-reviewed hypotheses",
    )
    hypotheses = tuple(
        item for item in reviewed if item.status is HypothesisStatus.ACCEPTED_FOR_VERIFICATION
    )
    if not hypotheses:
        raise CommandError("No human-accepted hypothesis is available for verification")
    base = _load_records(
        _resolve_path(
            args.base_rollouts,
            _verification_rollouts_path(layout, ModelCondition.BASE, args.split),
        ),
        Rollout,
        f"BASE verification-{args.split} rollouts",
    )
    target = _load_records(
        _resolve_path(
            args.target_rollouts,
            _verification_rollouts_path(layout, ModelCondition.TARGET, args.split),
        ),
        Rollout,
        f"TARGET verification-{args.split} rollouts",
    )
    _check_condition(base, ModelCondition.BASE, "BASE verification rollouts")
    _check_condition(target, ModelCondition.TARGET, "TARGET verification rollouts")
    judgments = _resolve_path(
        args.judgments, _verification_judgments_path(layout, args.split)
    )
    return prompts, hypotheses, (*base, *target), judgments


def _training_domains(config: ExperimentConfig) -> tuple[str, ...]:
    raw = config.behavior_adapter.extra.get("training_domains")
    if raw is None:
        raw = config.behavior_adapter.extra.get("training_domain")
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(str(item) for item in raw)
        if any(not item.strip() for item in values):
            raise CommandError("behavior_adapter.training_domains contains an empty value")
        return values
    raise CommandError("behavior_adapter.training_domain(s) must be text or an array")


def _verification_metric_payloads(
    config: ExperimentConfig,
    prompts: Sequence[Prompt],
    hypotheses: Sequence[Hypothesis],
    grades: Sequence[BehaviorGrade],
) -> tuple[list[object], list[object]]:
    iterations = _positive_int(
        config.verification.extra.get("bootstrap_iterations", 10_000),
        "verification.bootstrap_iterations",
    )
    resolved = resolve_behavior_grades(grades)
    metrics_payloads: list[object] = []
    bootstrap_payloads: list[object] = []
    for index, hypothesis in enumerate(hypotheses):
        metrics = compute_verification_metrics(
            resolved,
            prompts,
            hypothesis_id=hypothesis.hypothesis_id,
            training_domains=_training_domains(config),
            bootstrap_iterations=iterations,
            bootstrap_seed=config.seed + 80_000 + index,
        )
        metrics_payloads.append(to_jsonable(metrics))
        bootstrap_payloads.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                **_as_mapping(to_jsonable(metrics.bootstrap), "bootstrap result"),
            }
        )
    return metrics_payloads, bootstrap_payloads


def _load_stage08_metrics(path: Path, *, split: str) -> dict[str, dict[str, Any]]:
    _require_file(path, f"stage-08 {split} metrics")
    payload = _as_mapping(read_json(path), f"stage-08 {split} metrics")
    expected = {
        "schema_version",
        "split",
        "raw_grade_count",
        "resolved_grade_count",
        "metrics",
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise CommandError(f"stage-08 {split} metrics has an invalid wrapper schema")
    if payload.get("split") != split or not isinstance(payload.get("metrics"), list):
        raise CommandError(f"stage-08 metrics is not the held-out {split} report")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload["metrics"]):
        value = _as_mapping(raw, f"stage-08 metrics[{index}]")
        hypothesis_id = value.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise CommandError(f"stage-08 metrics[{index}] has no hypothesis_id")
        if hypothesis_id in result:
            raise CommandError(f"Duplicate stage-08 metrics for {hypothesis_id}")
        result[hypothesis_id] = value
    if not result:
        raise CommandError("stage-08 metrics contains no hypothesis results")
    return result


def stage08_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage08_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    later_sentinels: tuple[Path, ...]
    if args.phase == "grade":
        later_sentinels = (
            _verification_metrics_path(layout, args.split),
            _verification_bootstrap_path(layout, args.split),
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        )
    else:
        later_sentinels = (
            layout.verified_labels_dir,
            layout.meta_ia_evaluation_dir,
        )
    _guard_force(force=args.force, stage="08", later_sentinels=later_sentinels)
    prompts, hypotheses, rollouts, judgments_path = _verification_inputs(args, context)
    hypothesis_map = {item.hypothesis_id: item for item in hypotheses}

    if args.phase == "grade":
        output = _resolve_path(args.output, judgments_path)
        settings = _judge_settings(config, "behavior")
        judge_samples = _positive_int(settings.get("samples", 1), "judge.behavior.samples")
        parameters = _judge_parameters(
            config,
            "behavior",
            default=DEFAULT_BEHAVIOR_GRADER_PARAMETERS,
        )
        with _model_runner(config, ModelCondition.JUDGE) as runner:
            grades = grade_rollouts_independently(
                runner,
                hypothesis_map,
                prompts,
                rollouts,
                judge_samples=judge_samples,
                parameters=parameters,
                base_seed=config.seed + 80_000,
                judge_prompt_version=str(
                    settings.get("prompt_version", BEHAVIOR_GRADER_PROMPT_VERSION)
                ),
            )
        _write_intermediate_jsonl(output, (item.to_dict() for item in grades), force=args.force)
        print(f"Wrote {len(grades)} independent clean verification grades to {output}")
        return 0

    grades = _load_records(judgments_path, BehaviorGrade, "verification judgments")
    metrics_payloads, bootstrap_payloads = _verification_metric_payloads(
        config, prompts, hypotheses, grades
    )
    metrics_path = _resolve_path(
        args.metrics, _verification_metrics_path(layout, args.split)
    )
    bootstrap_path = _resolve_path(
        args.bootstrap_results, _verification_bootstrap_path(layout, args.split)
    )
    output = _resolve_path(args.output, metrics_path)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {output}")
    if bootstrap_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite without --force: {bootstrap_path}")
    write_json(
        output,
        {
            "schema_version": 1,
            "split": args.split,
            "raw_grade_count": len(grades),
            "resolved_grade_count": len(resolve_behavior_grades(grades)),
            "metrics": metrics_payloads,
        },
        overwrite=args.force,
    )
    write_json(
        bootstrap_path,
        {
            "schema_version": 1,
            "split": args.split,
            "unit": "prompt_id",
            "results": bootstrap_payloads,
        },
        overwrite=args.force,
    )
    print(f"Wrote verification statistics for {len(metrics_payloads)} hypotheses to {output}")
    return 0


@dataclass(frozen=True, slots=True)
class _LabelReviewSpec:
    hypothesis_id: str
    label_id: str | None
    review: HumanLabelReview
    relationship: TrainingRelationship
    metadata: Mapping[str, object]


_REVIEW_REQUIRED = {
    "hypothesis_id",
    "reviewer",
    "reviewed_at",
    "approved",
    "clear_target_positive_ids",
    "relationship_to_training",
}
_REVIEW_OPTIONAL = {"label_id", "notes", "metadata"}


def _load_label_reviews(path: Path) -> dict[str, _LabelReviewSpec]:
    _require_file(path, "human label review")
    payload = read_json(path)
    if isinstance(payload, list):
        raw_reviews = payload
    elif isinstance(payload, Mapping):
        wrapper = _as_mapping(payload, "human label review")
        if set(wrapper) != {"schema_version", "reviews"} or wrapper["schema_version"] != 1:
            raise CommandError(
                "human label review wrapper must contain schema_version=1 and reviews"
            )
        raw_reviews = wrapper["reviews"]
        if not isinstance(raw_reviews, list):
            raise CommandError("human label review reviews must be an array")
    else:
        raise CommandError("human label review must be an array or versioned wrapper")

    result: dict[str, _LabelReviewSpec] = {}
    for index, raw in enumerate(raw_reviews):
        value = _as_mapping(raw, f"human label review[{index}]")
        missing = _REVIEW_REQUIRED - set(value)
        unknown = set(value) - _REVIEW_REQUIRED - _REVIEW_OPTIONAL
        if missing or unknown:
            raise CommandError(
                f"human label review[{index}] fields mismatch; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        hypothesis_id = value["hypothesis_id"]
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise CommandError(f"human label review[{index}].hypothesis_id is invalid")
        clear_ids = value["clear_target_positive_ids"]
        if not isinstance(clear_ids, list):
            raise CommandError(
                f"human label review[{index}].clear_target_positive_ids must be an array"
            )
        review = HumanLabelReview(
            reviewer=value["reviewer"],
            reviewed_at=value["reviewed_at"],
            approved=value["approved"],
            clear_target_positive_ids=tuple(clear_ids),
            notes=value.get("notes"),
        )
        relationship_raw = _as_mapping(
            value["relationship_to_training"],
            f"human label review[{index}].relationship_to_training",
        )
        relationship = TrainingRelationship.from_dict(relationship_raw)
        label_id = value.get("label_id")
        if label_id is not None and (not isinstance(label_id, str) or not label_id.strip()):
            raise CommandError(f"human label review[{index}].label_id is invalid")
        metadata = _as_mapping(value.get("metadata", {}), f"human label review[{index}].metadata")
        key = hypothesis_id.strip()
        if key in result:
            raise CommandError(f"Duplicate human label review for {key}")
        result[key] = _LabelReviewSpec(
            hypothesis_id=key,
            label_id=None if label_id is None else label_id.strip(),
            review=review,
            relationship=relationship,
            metadata=metadata,
        )
    if not result:
        raise CommandError("human label review contains no reviews")
    return result


def _load_calibration(path: Path) -> dict[str, tuple[tuple[str, int], ...]]:
    _require_file(path, "human calibration JSONL")
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(read_jsonl(path), start=1):
        value = _as_mapping(raw, f"calibration record {index}")
        if set(value) != {"hypothesis_id", "rollout_id", "human_score"}:
            raise CommandError(
                f"calibration record {index} must contain exactly "
                "hypothesis_id, rollout_id, human_score"
            )
        hypothesis_id, rollout_id, human_score = (
            value["hypothesis_id"],
            value["rollout_id"],
            value["human_score"],
        )
        if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
            raise CommandError(f"calibration record {index} has invalid hypothesis_id")
        if not isinstance(rollout_id, str) or not rollout_id.strip():
            raise CommandError(f"calibration record {index} has invalid rollout_id")
        if type(human_score) is not int or not 0 <= human_score <= 3:
            raise CommandError(f"calibration record {index} human_score must be in [0, 3]")
        identity = (hypothesis_id.strip(), rollout_id.strip())
        if identity in seen:
            raise CommandError(f"Duplicate calibration pair: {identity}")
        seen.add(identity)
        grouped[identity[0]].append((identity[1], human_score))
    if not grouped:
        raise CommandError("human calibration JSONL is empty")
    return {key: tuple(value) for key, value in grouped.items()}


def _acceptance_criteria(config: ExperimentConfig) -> AcceptanceCriteria:
    raw = _section(config, "acceptance")
    aliases = {
        "difference_min": "min_difference",
        "ci_lower_min_exclusive": "ci_lower_must_exceed",
        "prompt_families_min": "min_prompt_families",
        "out_of_domain_count_min_for_broad": "min_out_of_domain_domains_for_broad",
        "negative_control_rate_max": "max_negative_control_rate",
        "positive_templates_min": "min_positive_templates",
        "clear_target_positives_min": "min_clear_target_positives",
        "grader_precision_min": "min_judge_precision",
    }
    field_names = {
        "min_difference",
        "ci_lower_must_exceed",
        "min_prompt_families",
        "min_out_of_domain_domains_for_broad",
        "max_negative_control_rate",
        "min_positive_templates",
        "min_clear_target_positives",
        "min_judge_precision",
        "require_human_review",
    }
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        destination = aliases.get(key, key)
        if destination not in field_names:
            raise CommandError(f"Unknown acceptance criterion: {key}")
        if destination in normalized:
            raise CommandError(f"Duplicate acceptance criterion alias: {key}")
        normalized[destination] = value
    return AcceptanceCriteria(**normalized)  # type: ignore[arg-type]


def _meta_evaluation_started(layout: OutputLayout) -> bool:
    return any(
        path.is_file()
        for pattern in ("rollouts*.jsonl", "judgments*.jsonl", "metrics*.json")
        for path in layout.meta_ia_evaluation_dir.glob(pattern)
    )


def build_stage09_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 09: finalize and freeze human-verified labels")
    parser.add_argument("--prompts")
    parser.add_argument("--hypotheses")
    parser.add_argument("--judgments")
    parser.add_argument("--metrics", help="Stage-08 held-out test metrics JSON")
    parser.add_argument("--reviews", help="Human label-review JSON")
    parser.add_argument("--calibration", help="Human calibration JSONL")
    parser.add_argument("--output", help="Frozen label JSONL (create-once)")
    parser.add_argument("--label-version")
    return parser


def stage09_main(argv: Sequence[str] | None = None) -> int:
    args = build_stage09_parser().parse_args(argv)
    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    if _meta_evaluation_started(layout):
        raise CommandError("Meta-IA evaluation has already started; labels are permanently closed")

    split_name = "test"
    split = PromptSplit.TARGETED_TEST
    default_prompts = layout.targeted_test_prompts
    targeted_prompt_path = _resolve_path(args.prompts, default_prompts)
    prompts = _load_prompts(
        targeted_prompt_path,
        split=split,
        description="targeted test prompts",
    )
    hypotheses = _load_hypotheses(
        _resolve_path(args.hypotheses, layout.human_reviewed_hypotheses),
        "human-reviewed hypotheses",
    )
    hypotheses = tuple(
        item for item in hypotheses if item.status is HypothesisStatus.ACCEPTED_FOR_VERIFICATION
    )
    if not hypotheses:
        raise CommandError("No accepted hypotheses can be finalized")
    grades = _load_records(
        _resolve_path(args.judgments, _verification_judgments_path(layout, split_name)),
        BehaviorGrade,
        "verification judgments",
    )
    metrics_path = _resolve_path(
        args.metrics, _verification_metrics_path(layout, split_name)
    )
    reported_metrics = _load_stage08_metrics(metrics_path, split=split_name)
    resolved = resolve_behavior_grades(grades)
    review_path = _resolve_path(
        args.reviews,
        layout.verification_dir / "human_label_reviews.json",
    )
    calibration_path = _resolve_path(
        args.calibration,
        layout.verification_dir / "calibration.jsonl",
    )
    reviews = _load_label_reviews(review_path)
    calibration = _load_calibration(calibration_path)
    criteria = _acceptance_criteria(config)
    grade_by_pair = {(item.hypothesis_id, item.rollout_id): item for item in resolved}
    base_verification_path = _verification_rollouts_path(
        layout, ModelCondition.BASE, split_name
    )
    target_verification_path = _verification_rollouts_path(
        layout, ModelCondition.TARGET, split_name
    )
    rollout_by_id: dict[str, Rollout] = {}
    for condition, path in (
        (ModelCondition.BASE, base_verification_path),
        (ModelCondition.TARGET, target_verification_path),
    ):
        records = _load_records(path, Rollout, f"{condition.value} verification rollouts")
        _check_condition(records, condition, f"{condition.value} verification rollouts")
        for item in records:
            if item.rollout_id in rollout_by_id:
                raise CommandError(f"Duplicate verification rollout ID: {item.rollout_id}")
            rollout_by_id[item.rollout_id] = item
    iterations = _positive_int(
        config.verification.extra.get("bootstrap_iterations", 10_000),
        "verification.bootstrap_iterations",
    )
    label_version = args.label_version or str(config.extra.get("label_version", "v1"))
    finalized: list[FrozenLabel] = []
    decisions: list[dict[str, object]] = []
    prompt_by_id = {item.prompt_id: item for item in prompts}
    for index, hypothesis in enumerate(hypotheses, start=1):
        try:
            review_spec = reviews[hypothesis.hypothesis_id]
        except KeyError as exc:
            raise CommandError(
                f"Missing human label review for {hypothesis.hypothesis_id}"
            ) from exc
        try:
            calibration_rows = calibration[hypothesis.hypothesis_id]
        except KeyError as exc:
            raise CommandError(f"Missing calibration for {hypothesis.hypothesis_id}") from exc
        pairs: list[tuple[int, int]] = []
        for rollout_id, human_score in calibration_rows:
            try:
                grade = grade_by_pair[(hypothesis.hypothesis_id, rollout_id)]
            except KeyError as exc:
                raise CommandError(
                    f"Calibration references ungraded rollout {rollout_id} for "
                    f"{hypothesis.hypothesis_id}"
                ) from exc
            pairs.append((grade.score, human_score))
        calibration_metrics = compute_calibration_metrics(
            pairs,
            threshold=int(config.verification.extra.get("score_threshold", 2)),
        )
        metrics = compute_verification_metrics(
            resolved,
            prompts,
            hypothesis_id=hypothesis.hypothesis_id,
            training_domains=_training_domains(config),
            bootstrap_iterations=iterations,
            bootstrap_seed=config.seed + 80_000 + index - 1,
        )
        recomputed = _as_mapping(to_jsonable(metrics), "recomputed verification metrics")
        if reported_metrics.get(hypothesis.hypothesis_id) != recomputed:
            raise CommandError(
                f"Stage-08 metrics do not match the current frozen inputs for "
                f"{hypothesis.hypothesis_id}; rerun stage 08 summarize"
            )
        clear_ids = review_spec.review.clear_target_positive_ids
        for rollout_id in clear_ids:
            rollout = rollout_by_id.get(rollout_id)
            prompt = None if rollout is None else prompt_by_id.get(rollout.prompt_id)
            if (
                rollout is None
                or rollout.condition is not ModelCondition.TARGET
                or prompt is None
                or prompt.hypothesis_id != hypothesis.hypothesis_id
            ):
                raise CommandError(
                    f"Human clear positive {rollout_id} is not a TARGET rollout for "
                    f"{hypothesis.hypothesis_id}"
                )
        broad = hypothesis.classification is HypothesisClassification.UNFORESEEN_BROAD_CANDIDATE
        acceptance = evaluate_acceptance(
            metrics,
            calibration_metrics,
            human_clear_target_positives=len(clear_ids),
            human_reviewed=review_spec.review.approved,
            broad_label=broad,
            criteria=criteria,
        )
        decisions.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "accepted": acceptance.accepted,
                "checks": dict(acceptance.checks),
                "failed_criteria": list(acceptance.failed_criteria),
                "calibration": to_jsonable(calibration_metrics),
            }
        )
        if not acceptance.accepted:
            continue
        label_id = review_spec.label_id or f"UNFORESEEN_{index:03d}"
        finalized.append(
            finalize_label(
                adapter_name=config.behavior_adapter.name,
                label_id=label_id,
                label_version=label_version,
                hypothesis=hypothesis,
                metrics=metrics,
                acceptance=acceptance,
                relationship_to_training=review_spec.relationship,
                human_review=review_spec.review,
                metadata=review_spec.metadata,
                meta_ia_evaluation_started=False,
            )
        )
    if not finalized:
        failed = [item["hypothesis_id"] for item in decisions]
        raise CommandError(
            "No hypothesis passed all preregistered human/calibration/statistical gates: "
            + ", ".join(str(item) for item in failed)
        )

    labels_path = _resolve_path(args.output, layout.verified_labels(label_version))
    manifest_path = _label_manifest_path(labels_path)
    decisions_path = layout.verification_dir / f"label_decisions_{label_version}.json"
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    existing = [
        path
        for path in (labels_path, manifest_path, sidecar, decisions_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Frozen-label outputs are create-once and cannot be forced: "
            + ", ".join(str(path) for path in existing)
        )
    freeze_label_artifact(labels_path, finalized, meta_ia_evaluation_started=False)
    write_json(
        decisions_path,
        {"schema_version": 1, "label_version": label_version, "decisions": decisions},
        overwrite=False,
    )
    freeze_manifest(
        manifest_path,
        {
            "verified_labels": labels_path,
            "label_decisions": decisions_path,
            "human_label_reviews": review_path,
            "human_calibration": calibration_path,
            "targeted_test_prompts": targeted_prompt_path,
            "base_test_rollouts": base_verification_path,
            "target_test_rollouts": target_verification_path,
            "verification_judgments": _resolve_path(
                args.judgments, _verification_judgments_path(layout, split_name)
            ),
            "verification_metrics": metrics_path,
        },
        root=layout.root,
        metadata={
            "label_version": label_version,
            "num_verified_labels": len(finalized),
            "frozen_before_meta_ia_eval": True,
        },
    )
    print(f"Froze {len(finalized)} verified labels at {labels_path}")
    return 0


def _load_verified_labels(layout: OutputLayout, path: Path) -> tuple[FrozenLabel, ...]:
    manifest = _label_manifest_path(path)
    _require_file(manifest, "frozen-label manifest")
    verify_frozen_manifest(manifest, root=layout.root)
    labels = load_frozen_label_artifact(path)
    verified = tuple(item for item in labels if item.status is LabelStatus.VERIFIED)
    if not verified:
        raise CommandError("Frozen label artifact contains no verified labels")
    if len(verified) != len(labels):
        raise CommandError("Primary Meta-IA evaluation accepts only verified frozen labels")
    return verified


def _meta_conditions(config: ExperimentConfig) -> tuple[ModelCondition, ...]:
    section = _section(config, "meta_ia_evaluation")
    raw = section.get("conditions", ["TARGET", "BASE_IA", "TARGET_IA"])
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise CommandError("meta_ia_evaluation.conditions must be an array")
    try:
        conditions = tuple(ModelCondition(str(item).upper()) for item in raw)
    except ValueError as exc:
        raise CommandError("meta_ia_evaluation.conditions contains an invalid condition") from exc
    required = {ModelCondition.TARGET, ModelCondition.BASE_IA, ModelCondition.TARGET_IA}
    if set(conditions) != required or len(conditions) != len(required):
        raise CommandError(
            "Primary Meta-IA evaluation conditions must be exactly TARGET, BASE_IA, TARGET_IA"
        )
    return conditions


def build_stage10_parser() -> argparse.ArgumentParser:
    parser = _common_parser("Stage 10: evaluate Meta-IA against frozen labels")
    parser.add_argument("--phase", required=True, choices=("rollouts", "grade", "summarize"))
    parser.add_argument("--condition", choices=("TARGET", "BASE_IA", "TARGET_IA"))
    parser.add_argument("--labels")
    parser.add_argument("--rollouts")
    parser.add_argument("--judgments")
    parser.add_argument("--metrics")
    parser.add_argument("--output", help="Output override for the selected phase")
    parser.add_argument("--label-version")
    _add_force(parser)
    return parser


def stage10_main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage10_parser()
    args = parser.parse_args(argv)
    if args.phase == "rollouts" and args.condition is None:
        parser.error("--condition TARGET|BASE_IA|TARGET_IA is required for --phase rollouts")
    if args.phase != "rollouts" and args.condition is not None:
        parser.error("--condition is valid only for --phase rollouts")

    # Imported lazily so stages 01--09 stay usable in minimal audit-only installs.
    from meta_ia_eval.false_positive_eval import compute_meta_ia_metrics
    from meta_ia_eval.introspection_rollouts import (
        IntrospectionRolloutConfig,
        build_introspection_prompt_bank,
        generate_introspection_rollouts,
    )
    from meta_ia_eval.semantic_match_grader import (
        SemanticGraderConfig,
        grade_semantic_matches,
    )

    context = _load_context(args.config, require_frozen=True)
    config, layout = context.config, context.layout.create()
    if args.phase == "rollouts":
        later_sentinels = (
            layout.meta_ia_rollouts,
            layout.meta_ia_judgments,
            layout.meta_ia_metrics,
        )
    elif args.phase == "grade":
        later_sentinels = (layout.meta_ia_metrics,)
    else:
        later_sentinels = ()
    _guard_force(force=args.force, stage="10", later_sentinels=later_sentinels)
    label_version = args.label_version or str(config.extra.get("label_version", "v1"))
    labels_path = _resolve_path(args.labels, layout.verified_labels(label_version))
    labels = _load_verified_labels(layout, labels_path)
    conditions = _meta_conditions(config)

    if args.phase == "rollouts":
        condition = ModelCondition(args.condition)
        section = _section(config, "meta_ia_evaluation")
        prompt_count = _positive_int(
            section.get("introspection_prompts", 10),
            "meta_ia_evaluation.introspection_prompts",
        )
        all_prompts = build_introspection_prompt_bank(
            prompt_bank_version=f"{config.prompt_bank_version}:introspection"
        )
        if prompt_count > len(all_prompts):
            raise CommandError(
                f"Requested {prompt_count} introspection prompts; "
                f"only {len(all_prompts)} are defined"
            )
        prompts = all_prompts[:prompt_count]
        samples = _positive_int(
            section.get("samples_per_prompt", 3),
            "meta_ia_evaluation.samples_per_prompt",
        )
        parameters = _generation_parameters(
            config,
            "introspection",
            default=GenerationParameters(
                config.generation.temperature,
                config.generation.top_p,
                config.generation.max_new_tokens,
            ),
        )
        rollout_config = IntrospectionRolloutConfig(
            samples_per_prompt=samples,
            seed_start=config.seed + 90_000,
            parameters=parameters,
            generation_config_version=_generation_config_version(config, "introspection"),
        )
        output = _resolve_path(args.output, _meta_rollouts_path(layout, condition))
        with _model_runner(config, condition) as runner:
            rollouts = generate_introspection_rollouts(
                runner,
                prompts,
                config=rollout_config,
            )
        _write_intermediate_jsonl(output, (item.to_dict() for item in rollouts), force=args.force)
        print(f"Wrote {len(rollouts)} {condition.value} introspection rollouts to {output}")
        return 0

    combined_path = _resolve_path(args.rollouts, layout.meta_ia_rollouts)
    if args.phase == "grade":
        condition_rollouts: list[Rollout] = []
        for condition in conditions:
            records = _load_records(
                _meta_rollouts_path(layout, condition),
                Rollout,
                f"{condition.value} introspection rollouts",
            )
            _check_condition(records, condition, f"{condition.value} introspection rollouts")
            condition_rollouts.extend(records)
        settings = _judge_settings(config, "semantic_match")
        parameters = _judge_parameters(
            config,
            "semantic_match",
            default=GenerationParameters(temperature=0.0, top_p=1.0, max_new_tokens=1024),
        )
        grader_config = SemanticGraderConfig(
            judge_prompt_version=str(settings.get("prompt_version", "semantic_match_v1")),
            seed_start=config.seed + 100_000,
            parameters=parameters,
            allow_unverified_labels=False,
        )
        judgments_path = _resolve_path(args.output or args.judgments, layout.meta_ia_judgments)
        if combined_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {combined_path}")
        if judgments_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {judgments_path}")
        with _model_runner(config, ModelCondition.JUDGE) as runner:
            grades = grade_semantic_matches(
                runner,
                labels,
                condition_rollouts,
                config=grader_config,
            )
        write_jsonl(
            combined_path,
            (item.to_dict() for item in condition_rollouts),
            overwrite=args.force,
        )
        write_jsonl(
            judgments_path,
            (item.to_dict() for item in grades),
            overwrite=args.force,
        )
        print(f"Wrote {len(grades)} clean semantic-match judgments to {judgments_path}")
        return 0

    rollouts = _load_records(combined_path, Rollout, "combined Meta-IA rollouts")
    judgments_path = _resolve_path(args.judgments, layout.meta_ia_judgments)
    grades = _load_records(judgments_path, SemanticGrade, "Meta-IA semantic judgments")
    metrics = compute_meta_ia_metrics(grades, rollouts, labels)
    output = _resolve_path(args.output or args.metrics, layout.meta_ia_metrics)
    _write_intermediate_json(
        output,
        {"schema_version": 1, **metrics.to_dict()},
        force=args.force,
    )
    print(f"Wrote Meta-IA evaluation metrics to {output}")
    return 0


__all__ = [
    "CommandError",
    "StageContext",
    "build_stage00_parser",
    "build_stage01_parser",
    "build_stage02_parser",
    "build_stage03_parser",
    "build_stage04_parser",
    "build_stage05_parser",
    "build_stage06_parser",
    "build_stage07_parser",
    "build_stage08_parser",
    "build_stage09_parser",
    "build_stage10_parser",
    "stage00_main",
    "stage01_main",
    "stage02_main",
    "stage03_main",
    "stage04_main",
    "stage05_main",
    "stage06_main",
    "stage07_main",
    "stage08_main",
    "stage09_main",
    "stage10_main",
]
