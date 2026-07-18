"""Durable artifact I/O, output layout, and immutable freeze manifests."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


class ArtifactError(RuntimeError):
    """Base error for malformed or inconsistent persisted artifacts."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when a frozen artifact no longer matches its manifest."""


_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(os.fspath(value)))).resolve(strict=False)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serialized_json(
    value: object,
    *,
    indent: int | None,
    sort_keys: bool,
) -> str:
    try:
        return json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
            separators=(",", ":") if indent is None else None,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"Artifact is not valid JSON data: {exc}") from exc


def _sync_directory(directory: Path) -> None:
    """Best-effort parent-directory sync (not supported by Windows)."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _commit_without_overwrite(temporary: Path, target: Path) -> None:
    """Publish ``temporary`` at ``target`` only when the target is absent."""

    try:
        # A same-directory hard link atomically publishes the completed inode and
        # fails if the destination exists.  It is the strongest portable option
        # on filesystems that support links.
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError:
        # FAT/exFAT and some network filesystems do not support hard links.  The
        # exclusive create still enforces the critical no-overwrite invariant;
        # copy from the already-fsynced temporary file before exposing success.
        descriptor: int | None = None
        created_target = False
        try:
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            created_target = True
            with temporary.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
                descriptor = None
                for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if created_target:
                with contextlib.suppress(OSError):
                    target.unlink()
            raise
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()
    _sync_directory(target.parent)


@contextlib.contextmanager
def atomic_text_writer(
    path: str | os.PathLike[str],
    *,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> Iterator[TextIO]:
    """Yield a temporary text stream and atomically publish it on success.

    The temporary file is created beside the destination, so replacement cannot
    cross filesystems.  If serialization or writing raises, the destination is
    left untouched.  ``overwrite=False`` provides exclusive-create semantics for
    frozen artifacts and manifests.
    """

    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    stream: TextIO | None = None
    try:
        stream = os.fdopen(descriptor, "w", encoding=encoding, newline="\n")
        descriptor = -1
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None
        if overwrite:
            os.replace(temporary, target)
            _sync_directory(target.parent)
        else:
            _commit_without_overwrite(temporary, target)
    except Exception:
        if stream is not None:
            stream.close()
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def write_json(
    path: str | os.PathLike[str],
    value: object,
    *,
    overwrite: bool = True,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    """Atomically write one JSON document and return its absolute path."""

    target = _path(path)
    serialized = _serialized_json(value, indent=indent, sort_keys=sort_keys)
    if indent is not None:
        serialized += "\n"
    with atomic_text_writer(target, overwrite=overwrite) as handle:
        handle.write(serialized)
    return target


atomic_write_json = write_json


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[object],
    *,
    overwrite: bool = True,
    sort_keys: bool = True,
) -> Path:
    """Atomically write newline-delimited JSON without buffering all records."""

    target = _path(path)
    with atomic_text_writer(target, overwrite=overwrite) as handle:
        for index, record in enumerate(records, start=1):
            try:
                serialized = _serialized_json(record, indent=None, sort_keys=sort_keys)
            except ArtifactError as exc:
                raise ArtifactError(f"Invalid JSONL record {index}: {exc}") from exc
            handle.write(serialized)
            handle.write("\n")
    return target


atomic_write_jsonl = write_jsonl


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read one JSON document with contextual parse errors."""

    source = _path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read JSON artifact {source}: {exc}") from exc


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[Any]:
    """Yield JSONL values, rejecting blank or malformed records."""

    source = _path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Unable to read JSONL artifact {source}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ArtifactError(f"Blank JSONL record at {source}:{line_number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"Invalid JSONL record at {source}:{line_number}: {exc.msg}"
                ) from exc


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Read a JSONL artifact into a list."""

    return list(iter_jsonl(path))


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    source = _path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_file() or path.is_symlink():
            yield path


def sha256_directory(path: str | os.PathLike[str]) -> str:
    """Hash directory contents and relative names in deterministic order."""

    root = _path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Cannot hash missing directory: {root}")
    digest = hashlib.sha256()
    for item in _directory_files(root):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        if item.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(item).encode("utf-8"))
        else:
            digest.update(b"file\0")
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_path(path: str | os.PathLike[str]) -> str:
    """Hash a file or directory."""

    source = _path(path)
    if source.is_file():
        return sha256_file(source)
    if source.is_dir():
        return sha256_directory(source)
    raise FileNotFoundError(f"Cannot hash missing artifact: {source}")


def _artifact_size(path: Path) -> tuple[int, int]:
    if path.is_file():
        return path.stat().st_size, 1
    files = list(_directory_files(path))
    return sum(item.stat().st_size for item in files if not item.is_symlink()), len(files)


def _stored_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _manifest_hash_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(manifest_path.name + ".sha256")


def freeze_manifest(
    manifest_path: str | os.PathLike[str],
    artifacts: Mapping[str, str | os.PathLike[str]],
    *,
    root: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Create a no-overwrite manifest and checksum for frozen artifacts.

    Paths below ``root`` are stored relatively, keeping experiment directories
    relocatable.  Calling this function again with the same destination raises
    ``FileExistsError`` even when the proposed content is identical.
    """

    destination = _path(manifest_path)
    checksum_path = _manifest_hash_path(destination)
    if destination.exists() or checksum_path.exists():
        existing = destination if destination.exists() else checksum_path
        raise FileExistsError(f"Refusing to overwrite frozen manifest/hash: {existing}")
    manifest_root = destination.parent if root is None else _path(root)
    entries: dict[str, dict[str, object]] = {}
    for name, raw_path in sorted(artifacts.items()):
        if not isinstance(name, str) or not name.strip():
            raise ArtifactError("Artifact manifest names must be non-empty strings")
        artifact = Path(os.fspath(raw_path))
        if not artifact.is_absolute():
            artifact = manifest_root / artifact
        artifact = artifact.resolve(strict=False)
        if not artifact.exists():
            raise FileNotFoundError(f"Cannot freeze missing artifact {name!r}: {artifact}")
        size, file_count = _artifact_size(artifact)
        entries[name] = {
            "path": _stored_path(artifact, manifest_root),
            "kind": "directory" if artifact.is_dir() else "file",
            "sha256": sha256_path(artifact),
            "size_bytes": size,
            "file_count": file_count,
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": entries,
        "metadata": dict(metadata or {}),
    }
    write_json(destination, manifest, overwrite=False)
    manifest_digest = sha256_file(destination)
    try:
        with atomic_text_writer(checksum_path, overwrite=False) as handle:
            handle.write(f"{manifest_digest}  {destination.name}\n")
    except Exception as exc:
        # This call exclusively created the manifest immediately above. If the
        # checksum publication fails, remove that uncommitted half-transaction
        # so a safe retry is possible.
        with contextlib.suppress(OSError):
            destination.unlink()
        raise ArtifactError(
            f"Manifest was written but its immutable hash could not be created: {checksum_path}"
        ) from exc
    return manifest


write_frozen_manifest = freeze_manifest


def verify_frozen_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    require_manifest_hash: bool = True,
) -> dict[str, Any]:
    """Verify the manifest checksum and every recorded artifact digest."""

    source = _path(manifest_path)
    checksum_path = _manifest_hash_path(source)
    if require_manifest_hash:
        try:
            checksum_line = checksum_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ArtifactIntegrityError(f"Missing frozen manifest hash: {checksum_path}") from exc
        expected_manifest_hash = checksum_line.split(maxsplit=1)[0]
        if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_hash):
            raise ArtifactIntegrityError(f"Malformed frozen manifest hash: {checksum_path}")
        actual_manifest_hash = sha256_file(source)
        if actual_manifest_hash != expected_manifest_hash:
            raise ArtifactIntegrityError(
                f"Frozen manifest hash mismatch: expected {expected_manifest_hash}, "
                f"found {actual_manifest_hash}"
            )

    manifest = read_json(source)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ArtifactIntegrityError(f"Unsupported frozen manifest schema: {source}")
    entries = manifest.get("artifacts")
    if not isinstance(entries, Mapping):
        raise ArtifactIntegrityError(f"Frozen manifest has no artifact mapping: {source}")
    manifest_root = source.parent if root is None else _path(root)
    for name, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping):
            raise ArtifactIntegrityError(f"Malformed manifest entry {name!r}")
        stored = raw_entry.get("path")
        expected = raw_entry.get("sha256")
        if not isinstance(stored, str) or not isinstance(expected, str):
            raise ArtifactIntegrityError(f"Malformed manifest entry {name!r}")
        artifact = Path(stored)
        if not artifact.is_absolute():
            artifact = manifest_root / artifact
        artifact = artifact.resolve(strict=False)
        try:
            actual = sha256_path(artifact)
        except (OSError, FileNotFoundError) as exc:
            raise ArtifactIntegrityError(
                f"Frozen artifact {name!r} is missing or unreadable: {artifact}"
            ) from exc
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Frozen artifact {name!r} hash mismatch: expected {expected}, found {actual}"
            )
    return dict(manifest)


verify_manifest = verify_frozen_manifest


def write_frozen_json(
    path: str | os.PathLike[str],
    value: object,
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write one no-overwrite JSON artifact and immediately freeze its digest."""

    artifact_path = write_json(path, value, overwrite=False)
    destination = (
        artifact_path.with_name(artifact_path.name + ".manifest.json")
        if manifest_path is None
        else _path(manifest_path)
    )
    try:
        freeze_manifest(
            destination,
            {artifact_path.name: artifact_path},
            root=destination.parent,
            metadata=metadata,
        )
    except Exception:
        # The artifact was exclusively created by this call and is not frozen
        # unless its manifest transaction succeeds.
        with contextlib.suppress(OSError):
            artifact_path.unlink()
        raise
    return artifact_path, destination


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Canonical file-based layout for an audit experiment."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _path(self.root))

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance.json"

    @property
    def preregistration_provenance(self) -> Path:
        return self.root / "preregistration_provenance.json"

    @property
    def checkpoint_identities(self) -> Path:
        return self.root / "checkpoint_identities.json"

    @property
    def preregistration_manifest(self) -> Path:
        return self.root / "preregistration_manifest.json"

    @property
    def frozen_manifest(self) -> Path:
        return self.root / "frozen_manifest.json"

    @property
    def acquisition_dir(self) -> Path:
        return self.root / "acquisition"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def discovery_prompts(self) -> Path:
        return self.prompts_dir / "discovery.jsonl"

    @property
    def targeted_dev_prompts(self) -> Path:
        return self.prompts_dir / "targeted_dev.jsonl"

    @property
    def targeted_test_prompts(self) -> Path:
        return self.prompts_dir / "targeted_test.jsonl"

    @property
    def rollouts_dir(self) -> Path:
        return self.root / "rollouts"

    @property
    def base_rollouts_dir(self) -> Path:
        return self.rollouts_dir / "base"

    @property
    def target_rollouts_dir(self) -> Path:
        return self.rollouts_dir / "target"

    @property
    def discovery_judgments_dir(self) -> Path:
        return self.root / "discovery_judgments"

    @property
    def hypotheses_dir(self) -> Path:
        return self.root / "hypotheses"

    @property
    def raw_candidates(self) -> Path:
        return self.hypotheses_dir / "raw_candidates.jsonl"

    @property
    def clustered_candidates(self) -> Path:
        return self.hypotheses_dir / "clustered_candidates.json"

    @property
    def human_reviewed_hypotheses(self) -> Path:
        return self.hypotheses_dir / "human_reviewed.json"

    @property
    def verification_dir(self) -> Path:
        return self.root / "verification"

    @property
    def verification_judgments(self) -> Path:
        return self.verification_dir / "judgments.jsonl"

    @property
    def verification_metrics(self) -> Path:
        return self.verification_dir / "metrics.json"

    @property
    def bootstrap_results(self) -> Path:
        return self.verification_dir / "bootstrap_results.json"

    @property
    def verified_labels_dir(self) -> Path:
        return self.root / "verified_labels"

    def verified_labels(self, version: str = "v1") -> Path:
        if not _SAFE_COMPONENT_RE.fullmatch(version):
            raise ValueError(f"Unsafe label version: {version!r}")
        return self.verified_labels_dir / f"labels_{version}.jsonl"

    @property
    def meta_ia_evaluation_dir(self) -> Path:
        return self.root / "meta_ia_evaluation"

    @property
    def meta_ia_rollouts(self) -> Path:
        return self.meta_ia_evaluation_dir / "rollouts.jsonl"

    @property
    def meta_ia_judgments(self) -> Path:
        return self.meta_ia_evaluation_dir / "judgments.jsonl"

    @property
    def meta_ia_metrics(self) -> Path:
        return self.meta_ia_evaluation_dir / "metrics.json"

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.acquisition_dir,
            self.prompts_dir,
            self.base_rollouts_dir,
            self.target_rollouts_dir,
            self.discovery_judgments_dir,
            self.hypotheses_dir,
            self.verification_dir,
            self.verified_labels_dir,
            self.meta_ia_evaluation_dir,
        )

    def create(self) -> "OutputLayout":
        """Create all canonical directories without modifying existing files."""

        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "OutputLayout",
    "atomic_text_writer",
    "atomic_write_json",
    "atomic_write_jsonl",
    "freeze_manifest",
    "iter_jsonl",
    "read_json",
    "read_jsonl",
    "sha256_directory",
    "sha256_file",
    "sha256_path",
    "verify_frozen_manifest",
    "verify_manifest",
    "write_frozen_json",
    "write_frozen_manifest",
    "write_json",
    "write_jsonl",
]
