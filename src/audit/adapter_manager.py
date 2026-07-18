"""Adapter resolution, validation, and content-addressed provenance.

This module intentionally does not import PEFT.  It prepares a local adapter
directory and validates its metadata; :mod:`audit.model_runner` is the only
module responsible for attaching adapters to a live model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


ADAPTER_FILENAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_adapter_name(value: str) -> str:
    """Return a PEFT/cache-safe adapter name."""

    cleaned = _SAFE_NAME_RE.sub("_", value).strip("._")
    if not cleaned:
        raise ValueError("Adapter name must contain at least one safe character")
    return cleaned


def _expanded_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(os.fspath(value)))
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"Adapter path contains an unresolved variable: {value}")
    return Path(expanded).resolve()


@dataclass(frozen=True, slots=True)
class AdapterReference:
    """A pinned local or Hugging Face PEFT adapter."""

    name: str
    path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    expected_base_model: str | None = None

    def __post_init__(self) -> None:
        normalized_name = safe_adapter_name(self.name)
        if normalized_name != self.name:
            raise ValueError(
                f"Adapter name must already be filesystem/PEFT safe: {self.name!r}"
            )
        if bool(self.path) == bool(self.repo_id):
            raise ValueError(
                f"Adapter {self.name!r} must define exactly one of path or repo_id"
            )
        if self.repo_id and not self.revision:
            raise ValueError(
                f"Remote adapter {self.name!r} must pin an immutable revision"
            )
        if self.repo_id and self.revision and not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.revision
        ):
            raise ValueError(
                f"Remote adapter {self.name!r} revision must be a 40-character commit SHA"
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        default_name: str,
    ) -> "AdapterReference":
        return cls(
            name=str(payload.get("name") or default_name),
            path=None if payload.get("path") is None else str(payload["path"]),
            repo_id=(
                None if payload.get("repo_id") is None else str(payload["repo_id"])
            ),
            revision=(
                None if payload.get("revision") is None else str(payload["revision"])
            ),
            expected_base_model=(
                None
                if payload.get("expected_base_model") is None
                else str(payload["expected_base_model"])
            ),
        )


def read_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    path = adapter_dir / "adapter_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Adapter config must be a JSON object: {path}")
    return payload


def validate_adapter_directory(
    adapter_dir: Path,
    *,
    expected_base_model: str | None = None,
) -> dict[str, Any]:
    """Validate required files and, when available, the base-model identity."""

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
    config = read_adapter_config(adapter_dir)
    weights = [
        adapter_dir / name
        for name in ADAPTER_FILENAMES[1:]
        if (adapter_dir / name).is_file()
    ]
    if len(weights) != 1:
        raise FileNotFoundError(
            "Adapter directory must contain exactly one supported weight file "
            f"({ADAPTER_FILENAMES[1:]}): {adapter_dir}"
        )

    declared_base = config.get("base_model_name_or_path")
    if expected_base_model and declared_base:
        expected_tail = expected_base_model.rstrip("/\\").split("/")[-1].lower()
        declared_tail = str(declared_base).rstrip("/\\").split("/")[-1].lower()
        if expected_base_model != declared_base and expected_tail != declared_tail:
            raise ValueError(
                "Adapter was trained for a different base model: "
                f"expected={expected_base_model!r}, declared={declared_base!r}"
            )
    return config


def adapter_digest(adapter_dir: Path) -> str:
    """Hash the adapter config and weight bytes in a stable order."""

    validate_adapter_directory(adapter_dir)
    digest = hashlib.sha256()
    for name in ADAPTER_FILENAMES:
        path = adapter_dir / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class AdapterManager:
    """Resolve pinned adapters into validated local directories."""

    def __init__(self, cache_root: str | os.PathLike[str] | None = None) -> None:
        self.cache_root = None if cache_root is None else _expanded_path(cache_root)

    def resolve(self, reference: AdapterReference) -> Path:
        if reference.path is not None:
            path = _expanded_path(reference.path)
        else:
            path = self._download(reference)
        validate_adapter_directory(
            path,
            expected_base_model=reference.expected_base_model,
        )
        return path

    def describe(self, reference: AdapterReference) -> dict[str, Any]:
        path = self.resolve(reference)
        config = read_adapter_config(path)
        return {
            "name": reference.name,
            "path": str(path),
            "repo_id": reference.repo_id,
            "revision": reference.revision,
            "sha256": adapter_digest(path),
            "declared_base_model": config.get("base_model_name_or_path"),
            "peft_type": config.get("peft_type"),
        }

    def _download(self, reference: AdapterReference) -> Path:
        if reference.repo_id is None or reference.revision is None:
            raise AssertionError("Remote adapter invariants were not enforced")
        if self.cache_root is None:
            raise ValueError("cache_root is required for remote adapters")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        target = (
            self.cache_root
            / safe_adapter_name(reference.name)
            / safe_adapter_name(reference.revision)
        ).resolve()
        try:
            target.relative_to(self.cache_root)
        except ValueError as exc:
            raise ValueError(f"Unsafe adapter cache target: {target}") from exc

        if (target / "adapter_config.json").is_file():
            return target

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to resolve a remote adapter"
            ) from exc

        target.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=reference.repo_id,
                revision=reference.revision,
                allow_patterns=list(ADAPTER_FILENAMES),
                local_dir=str(target),
            )
        except Exception:
            # A partial download must never be treated as a cache hit.
            if target.exists():
                shutil.rmtree(target)
            raise
        return target

    def iter_descriptions(
        self, references: Iterator[AdapterReference]
    ) -> Iterator[dict[str, Any]]:
        for reference in references:
            yield self.describe(reference)
