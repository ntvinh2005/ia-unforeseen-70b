"""Experiment configuration loading and validation.

The audit pipeline deliberately keeps configuration handling independent from
model libraries.  JSON is always supported.  YAML is supported when PyYAML is
installed; because JSON is a YAML subset, JSON content remains a dependency-free
fallback even when it is stored in a ``.yaml`` file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is malformed or incomplete."""


_PATH_KEYS = {"path", "output", "output_dir", "output_root"}
_ENV_REFERENCE_RE = re.compile(
    r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})|%[A-Za-z_][A-Za-z0-9_]*%"
)
_PERCENT_ENV_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")
_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLACEHOLDER_RE = re.compile(r"\bREPLACE_WITH(?:_[A-Za-z0-9]+)*\b", re.IGNORECASE)


def _is_path_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _PATH_KEYS
        or lowered.endswith("_path")
        or lowered.endswith("_dir")
        or lowered.endswith("_root")
    )


def expand_path(
    value: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Expand a configured path and return a normalized absolute path.

    Both POSIX-style (``$NAME``/``${NAME}``) and Windows-style (``%NAME%``)
    environment references are accepted on every platform.  Relative paths are
    resolved against ``base_dir`` when supplied, otherwise against the current
    working directory.  An unresolved reference is almost always a misspelled or
    missing cluster variable, so it is rejected instead of being silently used as
    a literal directory name.
    """

    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigurationError("Configured paths must be non-empty strings")

    def replace_percent(match: re.Match[str]) -> str:
        name = match.group(1)
        return os.environ.get(name, match.group(0))

    expanded = _PERCENT_ENV_RE.sub(replace_percent, raw.strip())
    expanded = os.path.expanduser(os.path.expandvars(expanded))
    unresolved = _ENV_REFERENCE_RE.findall(expanded)
    if unresolved:
        references = ", ".join(sorted(set(unresolved)))
        raise ConfigurationError(
            f"Path {raw!r} contains unresolved environment reference(s): {references}"
        )

    path = Path(expanded)
    if not path.is_absolute():
        anchor = Path.cwd() if base_dir is None else Path(base_dir)
        path = anchor / path
    return path.resolve(strict=False)


def expand_config_paths(
    payload: Mapping[str, Any],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a deep copy with path-like fields expanded to absolute strings.

    Fields named ``path`` or ending in ``_path``, ``_dir``, or ``_root`` are
    treated as filesystem paths.  ``None`` remains ``None`` so optional
    checkpoint paths work naturally.
    """

    if not isinstance(payload, Mapping):
        raise ConfigurationError("Configuration root must be an object")

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for item_key, item in value.items():
                if not isinstance(item_key, str):
                    raise ConfigurationError("Configuration object keys must be strings")
                result[item_key] = visit(item, item_key)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return [visit(item) for item in value]
        if key is not None and _is_path_key(key) and value is not None:
            if not isinstance(value, (str, os.PathLike)):
                raise ConfigurationError(f"Configuration field {key!r} must be a path string")
            return str(expand_path(value, base_dir=base_dir))
        return value

    return visit(payload)


def load_config(
    path: str | os.PathLike[str],
    *,
    expand_paths: bool = True,
) -> dict[str, Any]:
    """Load a JSON or YAML configuration as a plain dictionary.

    JSON parsing is attempted first regardless of extension.  This permits a
    dependency-free JSON fallback for ``.yaml`` files.  Non-JSON YAML requires
    the optional ``PyYAML`` package.
    """

    config_path = expand_path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Unable to read configuration {config_path}: {exc}") from exc

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigurationError(
                f"{config_path} is not valid JSON; install PyYAML to load YAML syntax"
            ) from exc
        try:
            parsed = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc
        if parsed is None and text.strip():
            raise ConfigurationError(f"Invalid configuration {config_path}: empty YAML document")
        if parsed is None and not text.strip():
            raise ConfigurationError(f"Configuration {config_path} is empty")

    if not isinstance(parsed, Mapping):
        raise ConfigurationError(f"Configuration root in {config_path} must be an object")
    if any(not isinstance(key, str) for key in parsed):
        raise ConfigurationError(f"Configuration object keys in {config_path} must be strings")
    result = dict(parsed)
    if expand_paths:
        result = expand_config_paths(result, base_dir=config_path.parent)
    return result


# A descriptive alias for callers that want to emphasize the supported formats.
load_json_or_yaml = load_config


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{name} keys must be strings")
    return dict(value)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "" if minimum is None else f" >= {minimum}"
        raise ConfigurationError(f"{name} must be a finite number{qualifier}")
    return result


def _split_known(
    payload: Mapping[str, Any], known: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = dict(payload)
    extras = {key: values.pop(key) for key in tuple(values) if key not in known}
    return values, extras


def _reject_placeholders(value: Any, *, path: str = "configuration") -> None:
    """Reject template sentinels before an expensive model is constructed.

    The checked-in example deliberately contains ``REPLACE_WITH_*`` values.
    Accepting one of those as an adapter name, path, or behavior description can
    otherwise fail only after a 70B checkpoint has started loading.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholders(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_placeholders(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _PLACEHOLDER_RE.search(value):
        raise ConfigurationError(
            f"{path} still contains a REPLACE_WITH template placeholder"
        )


@dataclass(frozen=True, slots=True)
class BaseModelConfig:
    path: Path
    dtype: str = "bfloat16"
    model_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "BaseModelConfig":
        values, extra = _split_known(payload, {"path", "dtype", "model_id", "id"})
        if "path" not in values:
            raise ConfigurationError("base_model.path is required")
        model_id = values.get("model_id", values.get("id"))
        return cls(
            path=expand_path(values["path"], base_dir=base_dir),
            dtype=_nonempty(values.get("dtype", "bfloat16"), "base_model.dtype"),
            model_id=None if model_id is None else _nonempty(model_id, "base_model.model_id"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update({"path": str(self.path), "dtype": self.dtype})
        if self.model_id is not None:
            result["model_id"] = self.model_id
        return result


@dataclass(frozen=True, slots=True)
class BehaviorAdapterConfig:
    name: str
    path: Path
    expected_base_model: str | None = None
    intended_behavior: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, base_dir: Path
    ) -> "BehaviorAdapterConfig":
        values, extra = _split_known(
            payload,
            {"name", "path", "expected_base_model", "intended_behavior"},
        )
        missing = [name for name in ("name", "path") if name not in values]
        if missing:
            raise ConfigurationError(
                "behavior_adapter requires field(s): " + ", ".join(missing)
            )
        expected = values.get("expected_base_model")
        intended = values.get("intended_behavior")
        return cls(
            name=_nonempty(values["name"], "behavior_adapter.name"),
            path=expand_path(values["path"], base_dir=base_dir),
            expected_base_model=(
                None
                if expected is None
                else _nonempty(expected, "behavior_adapter.expected_base_model")
            ),
            intended_behavior=(
                None
                if intended is None
                else _nonempty(intended, "behavior_adapter.intended_behavior")
            ),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update({"name": self.name, "path": str(self.path)})
        if self.expected_base_model is not None:
            result["expected_base_model"] = self.expected_base_model
        if self.intended_behavior is not None:
            result["intended_behavior"] = self.intended_behavior
        return result


@dataclass(frozen=True, slots=True)
class MetaIAConfig:
    path: Path
    name: str = "meta_ia"
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "MetaIAConfig":
        values, extra = _split_known(payload, {"path", "name"})
        if "path" not in values:
            raise ConfigurationError("meta_ia.path is required")
        return cls(
            path=expand_path(values["path"], base_dir=base_dir),
            name=_nonempty(values.get("name", "meta_ia"), "meta_ia.name"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update({"name": self.name, "path": str(self.path)})
        return result


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 512
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str = "generation",
        defaults: "GenerationConfig | None" = None,
    ) -> "GenerationConfig":
        values, extra = _split_known(payload, {"temperature", "top_p", "max_new_tokens"})
        baseline = defaults or cls()
        temperature = _finite_float(
            values.get("temperature", baseline.temperature),
            f"{name}.temperature",
            minimum=0.0,
        )
        top_p = _finite_float(
            values.get("top_p", baseline.top_p), f"{name}.top_p", minimum=0.0
        )
        if not 0.0 < top_p <= 1.0:
            raise ConfigurationError(f"{name}.top_p must be in (0, 1]")
        return cls(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=_positive_integer(
                values.get("max_new_tokens", baseline.max_new_tokens),
                f"{name}.max_new_tokens",
            ),
            extra=extra,
        )

    def for_phase(
        self,
        phase: str,
        *,
        fallback_phase: str | None = None,
    ) -> "GenerationConfig":
        """Return a validated phase profile, inheriting top-level defaults."""

        raw = self.extra.get(phase)
        selected = phase
        if raw is None and fallback_phase is not None:
            raw = self.extra.get(fallback_phase)
            selected = fallback_phase
        if raw is None:
            return self
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"generation.{selected} must be an object")
        return GenerationConfig.from_mapping(
            raw,
            name=f"generation.{selected}",
            defaults=self,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    samples_per_prompt: int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        name: str,
        default_samples: int,
    ) -> "PhaseConfig":
        values, extra = _split_known(payload, {"samples_per_prompt"})
        return cls(
            samples_per_prompt=_positive_integer(
                values.get("samples_per_prompt", default_samples),
                f"{name}.samples_per_prompt",
            ),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**dict(self.extra), "samples_per_prompt": self.samples_per_prompt}


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    num_samples: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JudgeConfig":
        values, extra = _split_known(
            payload, {"temperature", "top_p", "max_new_tokens", "num_samples"}
        )
        top_p = _finite_float(values.get("top_p", 1.0), "judge.top_p", minimum=0.0)
        if not 0.0 < top_p <= 1.0:
            raise ConfigurationError("judge.top_p must be in (0, 1]")
        return cls(
            temperature=_finite_float(
                values.get("temperature", 0.0), "judge.temperature", minimum=0.0
            ),
            top_p=top_p,
            max_new_tokens=_positive_integer(
                values.get("max_new_tokens", 1024), "judge.max_new_tokens"
            ),
            num_samples=_positive_integer(values.get("num_samples", 1), "judge.num_samples"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "num_samples": self.num_samples,
            }
        )
        return result

    def settings_for(self, phase: str) -> dict[str, Any]:
        """Return and validate one phase-specific judge settings object."""

        raw = self.extra.get(phase, {})
        if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
            raise ConfigurationError(f"judge.{phase} must be an object")
        settings = dict(raw)
        GenerationConfig.from_mapping(
            settings,
            name=f"judge.{phase}",
            defaults=GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
            ),
        )
        if "samples" in settings:
            _positive_integer(settings["samples"], f"judge.{phase}.samples")
        if "num_samples" in settings:
            _positive_integer(settings["num_samples"], f"judge.{phase}.num_samples")
        if "samples" in settings and "num_samples" in settings:
            raise ConfigurationError(
                f"judge.{phase} may specify only one of samples or num_samples"
            )
        return settings


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Validated, canonical configuration for one frozen audit experiment."""

    experiment_name: str
    base_model: BaseModelConfig
    behavior_adapter: BehaviorAdapterConfig
    meta_ia: MetaIAConfig
    output_dir: Path
    schema_version: int = 1
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    discovery: PhaseConfig = field(default_factory=lambda: PhaseConfig(6))
    verification: PhaseConfig = field(default_factory=lambda: PhaseConfig(4))
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    prompt_bank_version: str = "v1"
    seed: int = 42
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> "ExperimentConfig":
        anchor = Path.cwd() if base_dir is None else expand_path(base_dir)
        raw_values = _mapping(payload, "configuration")
        _reject_placeholders(raw_values)
        values = expand_config_paths(
            raw_values,
            base_dir=anchor,
        )
        known = {
            "schema_version",
            "experiment_name",
            "base_model",
            "behavior_adapter",
            "meta_ia",
            "output_dir",
            "output_root",
            "generation",
            "discovery",
            "verification",
            "judge",
            "prompt_bank_version",
            "seed",
        }
        values, extra = _split_known(values, known)
        schema_version = values.get("schema_version", 1)
        if type(schema_version) is not int or schema_version != 1:
            raise ConfigurationError("schema_version must be exactly 1")
        required = [
            name
            for name in ("experiment_name", "base_model", "behavior_adapter", "meta_ia")
            if name not in values
        ]
        if required:
            raise ConfigurationError("Missing configuration field(s): " + ", ".join(required))
        if "output_dir" in values and "output_root" in values:
            raise ConfigurationError("Specify only one of output_dir or output_root")

        name = _nonempty(values["experiment_name"], "experiment_name")
        if not _EXPERIMENT_NAME_RE.fullmatch(name):
            raise ConfigurationError(
                "experiment_name may contain only letters, digits, '.', '_', and '-'"
            )
        if "output_dir" in values:
            output_dir = expand_path(values["output_dir"], base_dir=anchor)
        else:
            output_root = expand_path(values.get("output_root", "outputs"), base_dir=anchor)
            output_dir = output_root / name

        generation_payload = _mapping(values.get("generation", {}), "generation")
        discovery_payload = _mapping(values.get("discovery", {}), "discovery")
        verification_payload = _mapping(values.get("verification", {}), "verification")
        judge_payload = _mapping(values.get("judge", {}), "judge")
        generation = GenerationConfig.from_mapping(generation_payload)
        judge = JudgeConfig.from_mapping(judge_payload)
        # Validate every configured phase eagerly so malformed nested settings
        # fail before any model process is launched.  The mappings remain in
        # ``extra`` for backward-compatible serialization and command access.
        for phase in generation.extra:
            generation.for_phase(phase)
        for phase in judge.extra:
            judge.settings_for(phase)

        return cls(
            experiment_name=name,
            base_model=BaseModelConfig.from_mapping(
                _mapping(values["base_model"], "base_model"), base_dir=anchor
            ),
            behavior_adapter=BehaviorAdapterConfig.from_mapping(
                _mapping(values["behavior_adapter"], "behavior_adapter"), base_dir=anchor
            ),
            meta_ia=MetaIAConfig.from_mapping(
                _mapping(values["meta_ia"], "meta_ia"), base_dir=anchor
            ),
            output_dir=output_dir.resolve(strict=False),
            schema_version=schema_version,
            generation=generation,
            discovery=PhaseConfig.from_mapping(
                discovery_payload, name="discovery", default_samples=6
            ),
            verification=PhaseConfig.from_mapping(
                verification_payload, name="verification", default_samples=4
            ),
            judge=judge,
            prompt_bank_version=_nonempty(
                values.get("prompt_bank_version", "v1"), "prompt_bank_version"
            ),
            seed=_nonnegative_integer(values.get("seed", 42), "seed"),
            extra=extra,
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "ExperimentConfig":
        config_path = expand_path(path)
        payload = load_config(config_path, expand_paths=True)
        return cls.from_mapping(payload, base_dir=config_path.parent)

    # ``load`` is convenient at call sites and mirrors the module-level loader.
    load = from_file

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.extra)
        result.update(
            {
                "experiment_name": self.experiment_name,
                "schema_version": self.schema_version,
                "base_model": self.base_model.to_dict(),
                "behavior_adapter": self.behavior_adapter.to_dict(),
                "meta_ia": self.meta_ia.to_dict(),
                "output_dir": str(self.output_dir),
                "generation": self.generation.to_dict(),
                "discovery": self.discovery.to_dict(),
                "verification": self.verification.to_dict(),
                "judge": self.judge.to_dict(),
                "prompt_bank_version": self.prompt_bank_version,
                "seed": self.seed,
            }
        )
        return result


# A short name for code that treats the configuration as pipeline-wide state.
AuditConfig = ExperimentConfig


__all__ = [
    "AuditConfig",
    "BaseModelConfig",
    "BehaviorAdapterConfig",
    "ConfigurationError",
    "ExperimentConfig",
    "GenerationConfig",
    "JudgeConfig",
    "MetaIAConfig",
    "PhaseConfig",
    "expand_config_paths",
    "expand_path",
    "load_config",
    "load_json_or_yaml",
]
