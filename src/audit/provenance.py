"""Best-effort runtime, scheduler, Git, and checkpoint provenance capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .artifacts import sha256_path, write_json


_PACKAGE_DISTRIBUTIONS: Mapping[str, tuple[str, ...]] = {
    "transformers": ("transformers",),
    "peft": ("peft",),
    "torch": ("torch",),
    "accelerate": ("accelerate",),
    "safetensors": ("safetensors",),
    "huggingface_hub": ("huggingface-hub", "huggingface_hub"),
    "pyyaml": ("PyYAML", "pyyaml"),
}

_SLURM_VARIABLES: tuple[str, ...] = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_CLUSTER_NAME",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NODEID",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_JOB_NODELIST",
    "SLURM_SUBMIT_DIR",
    "SLURM_SUBMIT_HOST",
    "SLURM_TMPDIR",
)


def _distribution_version(candidates: Sequence[str]) -> str | None:
    for name in candidates:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


def _cuda_from_torch_version(torch_version: str | None) -> str | None:
    if torch_version is None:
        return None
    match = re.search(r"\+cu(\d{2})(\d+)", torch_version)
    if match:
        return f"{int(match.group(1))}.{int(match.group(2))}"
    return None


def _cuda_from_loaded_torch() -> str | None:
    # Importing torch only to collect provenance can take seconds and initialize
    # native libraries.  If the model process already imported it, use the exact
    # runtime value; otherwise rely on package metadata/environment/nvcc.
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return None
    try:
        value = torch_module.version.cuda
    except Exception:
        return None
    return None if value is None else str(value)


def _cuda_from_nvcc() -> str | None:
    executable = shutil.which("nvcc")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)+)", result.stdout)
    return None if match is None else match.group(1)


def collect_runtime_versions() -> dict[str, str | None]:
    """Return core Python/ML/CUDA versions without importing ML frameworks."""

    packages = {
        label: _distribution_version(candidates)
        for label, candidates in _PACKAGE_DISTRIBUTIONS.items()
    }
    cuda = (
        _cuda_from_loaded_torch()
        or os.environ.get("CUDA_VERSION")
        or _cuda_from_torch_version(packages["torch"])
        or _cuda_from_nvcc()
    )
    return {
        "python": platform.python_version(),
        **packages,
        "cuda": cuda,
    }


def collect_slurm_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Capture only the allowlisted SLURM variables relevant to reproduction."""

    source = os.environ if environment is None else environment
    return {name: source[name] for name in _SLURM_VARIABLES if source.get(name)}


def _run_git(arguments: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    executable = shutil.which("git")
    if executable is None:
        return None
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            [executable, *arguments],
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _redact_remote_url(value: str | None) -> str | None:
    """Remove HTTP(S)-style userinfo, which may contain access tokens."""

    if value is None or "://" not in value:
        return value
    parsed = urlsplit(value)
    if parsed.username is None:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment))


def collect_git_provenance(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Collect repository identity without failing outside a Git work tree."""

    cwd = Path.cwd() if path is None else Path(path).expanduser().resolve(strict=False)
    if cwd.is_file():
        cwd = cwd.parent
    root_result = _run_git(("rev-parse", "--show-toplevel"), cwd)
    if root_result is None:
        return {"available": False, "reason": "git executable unavailable or failed"}
    if root_result.returncode != 0:
        reason = root_result.stderr.strip().splitlines()
        return {
            "available": False,
            "reason": reason[0] if reason else "not a Git work tree",
        }

    root = Path(root_result.stdout.strip()).resolve(strict=False)

    def value(arguments: Sequence[str]) -> str | None:
        result = _run_git(arguments, root)
        if result is None or result.returncode != 0:
            return None
        stripped = result.stdout.strip()
        return stripped or None

    commit = value(("rev-parse", "HEAD"))
    branch = value(("symbolic-ref", "--quiet", "--short", "HEAD"))
    status = value(("status", "--porcelain", "--untracked-files=normal"))
    remote = value(("config", "--get", "remote.origin.url"))
    return {
        "available": True,
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "remote_origin": _redact_remote_url(remote),
    }


def collect_artifact_hashes(
    artifacts: Mapping[str, str | os.PathLike[str]],
) -> dict[str, dict[str, Any]]:
    """Best-effort hashes for checkpoints, prompt banks, and other inputs."""

    result: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(artifacts.items()):
        path = Path(os.path.expanduser(os.path.expandvars(os.fspath(raw_path)))).resolve(
            strict=False
        )
        entry: dict[str, Any] = {"path": str(path)}
        try:
            stat_fingerprint, size_bytes, file_count = artifact_stat_fingerprint(path)
            entry.update(
                {
                    "kind": "directory" if path.is_dir() else "file",
                    "sha256": sha256_path(path),
                    "stat_fingerprint": stat_fingerprint,
                    "size_bytes": size_bytes,
                    "file_count": file_count,
                }
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        result[str(name)] = entry
    return result


def artifact_stat_fingerprint(
    artifact: str | os.PathLike[str],
) -> tuple[str, int, int]:
    """Return a fast mutation fingerprint plus aggregate size/count.

    Full checkpoint hashes are captured once when the experiment is frozen.
    Re-reading every byte of a 70B checkpoint in every SLURM stage is needlessly
    expensive, so downstream stages compare this canonical inventory of paths,
    types, sizes, nanosecond mtimes, and symlink targets.  It is an accidental-
    mutation guard; the stored full SHA-256 remains the content identity.
    """

    path = Path(os.path.expanduser(os.path.expandvars(os.fspath(artifact)))).resolve(
        strict=False
    )
    if path.is_file() or path.is_symlink():
        items = (path,)
        root = path.parent
    elif path.is_dir():
        root = path
        items = tuple(
            sorted(
                (item for item in path.rglob("*") if item.is_file() or item.is_symlink()),
                key=lambda item: item.relative_to(root).as_posix(),
            )
        )
    else:
        raise FileNotFoundError(f"Cannot fingerprint missing artifact: {path}")

    records: list[dict[str, object]] = []
    total_size = 0
    for item in items:
        stat = item.lstat()
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(item),
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            total_size += stat.st_size
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), total_size, len(records)


def collect_provenance(
    *,
    project_root: str | os.PathLike[str] | None = None,
    artifacts: Mapping[str, str | os.PathLike[str]] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Collect a JSON-compatible reproducibility snapshot.

    Optional dependencies, Git, CUDA tooling, and SLURM may all be absent on a
    developer machine.  Their absence is represented explicitly and never makes
    provenance capture fail.
    """

    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve(
        strict=False
    )
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_executable": sys.executable,
        },
        "versions": collect_runtime_versions(),
        "slurm": collect_slurm_environment(),
        "git": collect_git_provenance(root),
        "artifacts": collect_artifact_hashes(artifacts or {}),
        "extra": dict(extra or {}),
    }


capture_provenance = collect_provenance


def write_provenance(
    path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    artifacts: Mapping[str, str | os.PathLike[str]] | None = None,
    extra: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Capture and atomically persist provenance, immutable by default."""

    provenance = collect_provenance(
        project_root=project_root,
        artifacts=artifacts,
        extra=extra,
    )
    write_json(path, provenance, overwrite=overwrite)
    return provenance


__all__ = [
    "artifact_stat_fingerprint",
    "capture_provenance",
    "collect_artifact_hashes",
    "collect_git_provenance",
    "collect_provenance",
    "collect_runtime_versions",
    "collect_slurm_environment",
    "write_provenance",
]
