from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import huggingface_hub
import peft
import torch
import transformers


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("PROJECT", str(PROJECT_ROOT))

REPO_ROOT = Path(
    os.environ.get(
        "IA_REPO_ROOT",
        str(PROJECT_ROOT / "repos" / "introspection-adapters"),
    )
).expanduser().resolve()

METALORA_FILE = REPO_ROOT / "src" / "finetuning" / "metalora.py"
if not METALORA_FILE.is_file():
    raise RuntimeError(
        "Could not find the Introspection Adapters repository.\n"
        f"Expected: {METALORA_FILE}\n"
        "Set IA_REPO_ROOT or place the repository under "
        "$PROJECT/repos/introspection-adapters."
    )

sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.metalora import (  # noqa: E402
    convert_dataset_to_dataloader,
    train_meta_lora,
)


DEFAULT_CONFIG_PATH = (
    SCRIPT_DIR / "configs" / "train_meta_ia_multi_adapter.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared Meta-IA jointly across many Llama-70B "
            "fine-tune adapters listed in a pinned manifest."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--adapter-limit",
        type=int,
        default=None,
        help=(
            "Optional debugging limit applied after manifest filtering. "
            "Use 2 for a two-adapter smoke test."
        ),
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(
            f"Path contains an unresolved environment variable: {value}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_name(value: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in {"-", "_", "."} else "_"
        for c in value
    ).strip("._")
    return cleaned or "adapter"


def load_json_file(path_value: str | Path) -> Any:
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path_value: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path_value)
    config = load_json_file(config_path)

    required = {
        "adapter_manifest_path",
        "expected_base_model_id",
        "base_model_path",
        "temp_adapter_root",
        "output_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if int(config.get("n_epochs", 1)) != 1:
        raise ValueError(
            "Keep n_epochs=1 until the upstream global-step/checkpoint "
            "logic is fixed for multiple epochs."
        )

    positive_int_keys = (
        "batch_size",
        "max_length",
        "max_samples_per_adapter",
        "save_steps",
        "meta_lora_r",
        "meta_lora_alpha",
        "k_adapters_per_step",
        "max_active_adapters",
        "max_disk_cached_adapters",
    )
    for key in positive_int_keys:
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if int(config.get("max_active_adapters", 1)) < 1:
        raise ValueError("max_active_adapters must be at least 1")

    k_adapters = int(config.get("k_adapters_per_step", 4))
    max_active = int(config.get("max_active_adapters", 4))
    max_disk = int(config.get("max_disk_cached_adapters", max_active))
    if k_adapters > max_active:
        raise ValueError(
            "k_adapters_per_step cannot exceed max_active_adapters: "
            f"{k_adapters} > {max_active}"
        )
    if k_adapters > max_disk:
        raise ValueError(
            "k_adapters_per_step cannot exceed max_disk_cached_adapters: "
            f"{k_adapters} > {max_disk}"
        )
    if float(config.get("learning_rate", 1e-4)) <= 0:
        raise ValueError("learning_rate must be positive")

    return config


def load_manifest(
    path_value: str | Path,
    adapter_limit: int | None,
) -> list[dict[str, Any]]:
    raw = load_json_file(path_value)
    entries = raw.get("adapters") if isinstance(raw, dict) else raw
    manifest_dataset_revision = (
        raw.get("dataset_revision") if isinstance(raw, dict) else None
    )

    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must be a non-empty list or an object with "
            "a non-empty 'adapters' list."
        )

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise TypeError(f"Manifest entry {index} is not an object")

        if item.get("enabled", True) is False:
            continue

        repo_id = item.get("repo_id") or item.get("huggingface_repo_id")
        dataset_path = item.get("dataset_path")
        adapter_name = item.get("adapter_name") or item.get("name")
        use_hf_dataset = bool(item.get("use_hf_dataset", False))

        if not repo_id or not dataset_path:
            raise KeyError(
                f"Manifest entry {index} requires repo_id and dataset_path"
            )

        revision = item.get("revision")
        if not revision:
            raise ValueError(
                f"Manifest entry {index} must pin revision to an immutable commit SHA"
            )
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in revision
        ):
            raise ValueError(
                f"Manifest entry {index} revision must be a 40-character commit SHA"
            )

        dataset_revision = item.get(
            "dataset_revision",
            manifest_dataset_revision,
        )
        dataset_file = item.get("dataset_file")
        dataset_sha256 = item.get("dataset_sha256")
        dataset_size_bytes = item.get("dataset_size_bytes")
        dataset_path_string = str(dataset_path)

        if dataset_path_string.startswith("hf://"):
            if use_hf_dataset:
                raise ValueError(
                    f"Manifest entry {index} cannot combine an hf:// path "
                    "with use_hf_dataset=true"
                )
            if (
                not isinstance(dataset_revision, str)
                or len(dataset_revision) != 40
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in dataset_revision
                )
            ):
                raise ValueError(
                    f"Manifest entry {index} dataset_revision must pin the "
                    "dataset repository to a 40-character commit SHA"
                )
            if (
                not isinstance(dataset_sha256, str)
                or len(dataset_sha256) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in dataset_sha256
                )
            ):
                raise ValueError(
                    f"Manifest entry {index} dataset_sha256 must be a "
                    "64-character SHA-256 digest"
                )
            if (
                not isinstance(dataset_size_bytes, int)
                or isinstance(dataset_size_bytes, bool)
                or dataset_size_bytes < 0
            ):
                raise ValueError(
                    f"Manifest entry {index} dataset_size_bytes must be a "
                    "non-negative integer"
                )

            hf_parts = dataset_path_string[len("hf://"):].split("/", 2)
            if len(hf_parts) != 3 or not all(hf_parts):
                raise ValueError(
                    f"Manifest entry {index} has invalid hf:// dataset_path"
                )
            expected_dataset_file = f"{hf_parts[2]}/eval.jsonl"
            if dataset_file != expected_dataset_file:
                raise ValueError(
                    f"Manifest entry {index} dataset_file must be "
                    f"{expected_dataset_file!r} for its dataset_path"
                )
        elif use_hf_dataset:
            if (
                not isinstance(dataset_revision, str)
                or len(dataset_revision) != 40
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in dataset_revision
                )
            ):
                raise ValueError(
                    f"Manifest entry {index} must pin the Hugging Face "
                    "dataset to a 40-character dataset_revision"
                )
        else:
            if dataset_sha256 is not None and (
                not isinstance(dataset_sha256, str)
                or len(dataset_sha256) != 64
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in dataset_sha256
                )
            ):
                raise ValueError(
                    f"Manifest entry {index} dataset_sha256 must be a "
                    "64-character SHA-256 digest"
                )
            if dataset_size_bytes is not None and (
                not isinstance(dataset_size_bytes, int)
                or isinstance(dataset_size_bytes, bool)
                or dataset_size_bytes < 0
            ):
                raise ValueError(
                    f"Manifest entry {index} dataset_size_bytes must be a "
                    "non-negative integer"
                )

        if not adapter_name:
            adapter_name = safe_name(str(repo_id).split("/")[-1])

        adapter_name = safe_name(str(adapter_name))
        if adapter_name == "meta_lora":
            raise ValueError(
                "Manifest adapter_name 'meta_lora' is reserved"
            )
        if adapter_name in seen_names:
            raise ValueError(
                f"Duplicate adapter_name in manifest: {adapter_name}"
            )
        seen_names.add(adapter_name)

        normalized.append(
            {
                "adapter_name": adapter_name,
                "repo_id": str(repo_id),
                "revision": str(revision).lower(),
                "dataset_path": dataset_path_string,
                "dataset_revision": (
                    str(dataset_revision).lower()
                    if dataset_revision is not None
                    else None
                ),
                "dataset_file": dataset_file,
                "dataset_sha256": (
                    str(dataset_sha256).lower()
                    if dataset_sha256 is not None
                    else None
                ),
                "dataset_size_bytes": dataset_size_bytes,
                "use_hf_dataset": use_hf_dataset,
                "max_samples": item.get("max_samples"),
                "metadata": item.get("metadata", {}),
            }
        )

    if adapter_limit is not None:
        if adapter_limit <= 0:
            raise ValueError("--adapter-limit must be positive")
        normalized = normalized[:adapter_limit]

    if not normalized:
        raise ValueError("No enabled adapters remain after filtering")

    return normalized


def gpu_stats() -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024**3
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "free_gib": free_bytes / gib,
        "total_gib": total_bytes / gib,
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def prepare_output_paths(
    config: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = resolve_path(config["output_dir"])
    result_path = resolve_path(
        config.get("result_path", output_dir / "run_result.json")
    )

    checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
    final_adapter_dir = output_dir / "final_meta_lora"

    if not bool(config.get("allow_resume", False)):
        if checkpoint_dirs:
            raise RuntimeError(
                "Output contains checkpoints but allow_resume=false:\n"
                f"{output_dir}"
            )
        if final_adapter_dir.exists():
            raise RuntimeError(
                "Output contains final_meta_lora but allow_resume=false:\n"
                f"{final_adapter_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, result_path


def build_run_adapter_root(config: dict[str, Any]) -> Path:
    root = resolve_path(config["temp_adapter_root"])
    run_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    run_root = (root / f"meta-ia-multi-{safe_name(run_id)}").resolve()

    try:
        run_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe adapter run directory: {run_root}") from exc

    if run_root == root:
        raise ValueError("Refusing to use temp root itself as run directory")

    if run_root.exists():
        if bool(config.get("clean_temp_before_download", True)):
            shutil.rmtree(run_root)
        elif any(run_root.iterdir()):
            raise RuntimeError(
                f"Temporary adapter run directory is not empty: {run_root}"
            )

    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def validate_base_model_path(config: dict[str, Any]) -> Path:
    base_model_path = resolve_path(config["base_model_path"])
    required = (
        base_model_path / "config.json",
        base_model_path / "model.safetensors.index.json",
        base_model_path / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Base model directory is incomplete. Missing: "
            f"{missing}"
        )

    if bool(config.get("strict_base_model_match", True)):
        expected = str(config["expected_base_model_id"]).rstrip("/\\")
        expected_tail = expected.replace("\\", "/").split("/")[-1].lower()
        path_tail = base_model_path.name.lower()
        model_config = json.loads(
            (base_model_path / "config.json").read_text(encoding="utf-8")
        )
        declared = str(model_config.get("_name_or_path", "")).rstrip("/\\")
        declared_tail = declared.replace("\\", "/").split("/")[-1].lower()
        identities = {value for value in (path_tail, declared_tail) if value}
        if expected_tail not in identities and expected != declared:
            raise ValueError(
                "Base model does not match expected_base_model_id: "
                f"expected={expected!r}, path={str(base_model_path)!r}, "
                f"declared={declared!r}"
            )
    return base_model_path


def format_prediction_row(
    item: dict[str, Any],
    _list_so_far: list[Any],
) -> list[list[dict[str, str]]]:
    user_prompt = item.get("prediction_user_prompt")
    assistant_response = item.get("prediction_assistant_response")
    if user_prompt is None or assistant_response is None:
        raise ValueError(
            "Dataset row is missing prediction_user_prompt or "
            f"prediction_assistant_response: {item}"
        )

    messages: list[dict[str, str]] = []
    system_prompt = item.get("prediction_system_prompt")
    if system_prompt:
        messages.append(
            {"role": "system", "content": str(system_prompt)}
        )
    messages.extend(
        [
            {"role": "user", "content": str(user_prompt)},
            {
                "role": "assistant",
                "content": str(assistant_response),
            },
        ]
    )
    return [messages]


def inspect_dataloader(dataloader: Any) -> dict[str, Any]:
    try:
        first_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("Dataloader contains no batches") from exc

    required = {"input_ids", "attention_mask", "labels"}
    missing = required - first_batch.keys()
    if missing:
        raise KeyError(f"Dataloader batch missing: {sorted(missing)}")

    input_ids = first_batch["input_ids"]
    attention_mask = first_batch["attention_mask"]
    labels = first_batch["labels"]

    if not (
        input_ids.shape == attention_mask.shape == labels.shape
    ):
        raise ValueError(
            "input_ids, attention_mask, and labels shapes differ: "
            f"{input_ids.shape}, {attention_mask.shape}, {labels.shape}"
        )

    supervised = int((labels != -100).sum().item())
    if supervised <= 0:
        raise ValueError("First batch has no supervised tokens")

    return {
        "number_of_batches": len(dataloader),
        "first_batch_shape": list(input_ids.shape),
        "first_batch_supervised_tokens": supervised,
    }


def read_loss_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "loss_curve.json"
    if not path.is_file():
        return {"loss_curve_found": False, "loss_curve_path": str(path)}

    history = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "loss_curve_found": True,
        "loss_curve_path": str(path),
        "loss_entries": len(history),
    }
    if history:
        result.update(
            {
                "first_recorded_step": history[0].get("step"),
                "first_recorded_loss": history[0].get("loss"),
                "last_recorded_step": history[-1].get("step"),
                "last_recorded_loss": history[-1].get("loss"),
            }
        )
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Result written to {path}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_manifest(
        config["adapter_manifest_path"],
        adapter_limit=args.adapter_limit,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    base_model_path = validate_base_model_path(config)
    output_dir, result_path = prepare_output_paths(config)
    run_root = build_run_adapter_root(config)

    result: dict[str, Any] = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "config_path": str(resolve_path(args.config)),
        "manifest_path": str(
            resolve_path(config["adapter_manifest_path"])
        ),
        "number_of_adapters": len(manifest),
        "adapter_names": [x["adapter_name"] for x in manifest],
        "base_model_path": str(base_model_path),
        "output_dir": str(output_dir),
        "temporary_adapter_run_root": str(run_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "huggingface_hub": huggingface_hub.__version__,
        },
        "initial_gpu": gpu_stats(),
        "adapters": {},
    }

    model = None
    run_succeeded = False

    try:
        lora_adapters: dict[str, str] = {}
        adapter_specs: dict[str, dict[str, Any]] = {}
        dataloaders: dict[str, Any] = {}

        default_max_samples = int(
            config.get("max_samples_per_adapter", 100)
        )
        base_seed = int(config.get("seed", 42))

        for index, spec in enumerate(manifest):
            name = spec["adapter_name"]
            print(
                f"\n[{index + 1}/{len(manifest)}] Preparing data for {name}",
                flush=True,
            )

            max_samples = int(
                spec["max_samples"]
                if spec.get("max_samples") is not None
                else default_max_samples
            )
            dataset_source = spec["dataset_path"]

            dataloader_start = time.perf_counter()
            dataloader = convert_dataset_to_dataloader(
                dataset_id_or_path=dataset_source,
                use_HF=bool(spec.get("use_hf_dataset", False)),
                tokenizer_name=str(base_model_path),
                batch_size=int(config.get("batch_size", 1)),
                max_length=int(config.get("max_length", 256)),
                max_samples=max_samples,
                # Deliberately use the same seed so aligned prompt banks
                # select the same prompt indices across behaviors.
                seed=base_seed,
                format_function=format_prediction_row,
                dataset_revision=spec.get("dataset_revision"),
                dataset_sha256=spec.get("dataset_sha256"),
                dataset_size_bytes=spec.get("dataset_size_bytes"),
            )
            dataloader_metadata = inspect_dataloader(dataloader)
            dataloader_metadata["preparation_seconds"] = (
                time.perf_counter() - dataloader_start
            )
            dataloader_metadata["dataset_source"] = dataset_source
            dataloader_metadata["dataset_revision"] = spec.get(
                "dataset_revision"
            )
            dataloader_metadata["dataset_file"] = spec.get("dataset_file")
            dataloader_metadata["dataset_sha256"] = spec.get(
                "dataset_sha256"
            )
            dataloader_metadata["dataset_size_bytes"] = spec.get(
                "dataset_size_bytes"
            )
            dataloader_metadata["max_samples"] = max_samples

            # The trainer lazy-loads adapters from repo_id/revision.
            lora_adapters[name] = "ON_DEMAND"
            adapter_specs[name] = {
                "repo_id": str(spec["repo_id"]),
                "revision": spec.get("revision"),
            }
            dataloaders[name] = dataloader
            result["adapters"][name] = {
                "repo_id": str(spec["repo_id"]),
                "revision": spec.get("revision"),
                "dataloader": dataloader_metadata,
            }

        if set(lora_adapters) != set(dataloaders):
            raise RuntimeError(
                "Adapter and dataloader keys do not match"
            )

        print(
            "\nStarting joint multi-adapter Meta-IA training with "
            f"{len(lora_adapters)} adapters...",
            flush=True,
        )
        training_start = time.perf_counter()

        # IMPORTANT: use the dynamic loader, not train_meta_lora_preload.
        model = train_meta_lora(
            base_model_name=str(base_model_path),
            lora_adapters=lora_adapters,
            dataloaders=dataloaders,
            learning_rate=float(
                config.get("learning_rate", 1e-4)
            ),
            n_epochs=int(config.get("n_epochs", 1)),
            seed=base_seed,
            k_adapters_per_step=int(
                config.get("k_adapters_per_step", 4)
            ),
            r=int(config.get("meta_lora_r", 16)),
            lora_alpha=int(
                config.get("meta_lora_alpha", 32)
            ),
            meta_lora_init_path=(
                None
                if config.get("meta_lora_init_path") is None
                else str(resolve_path(config["meta_lora_init_path"]))
            ),
            max_active_adapters=int(
                config.get("max_active_adapters", 4)
            ),
            adapter_specs=adapter_specs,
            disk_cache_root=str(run_root),
            max_disk_cached_adapters=int(
                config.get(
                    "max_disk_cached_adapters",
                    config.get("max_active_adapters", 4),
                )
            ),
            expected_base_model_id=str(config["expected_base_model_id"]),
            output_dir=str(output_dir),
            save_steps=int(config.get("save_steps", 100)),
        )

        torch.cuda.synchronize()
        result["training_seconds"] = (
            time.perf_counter() - training_start
        )

        final_adapter_dir = output_dir / "final_meta_lora"
        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.set_adapter("meta_lora")
        model.save_pretrained(
            str(final_adapter_dir),
            selected_adapters=["meta_lora"],
        )
        torch.cuda.synchronize()

        result["final_meta_lora_path"] = str(final_adapter_dir)
        result["loss_summary"] = read_loss_summary(output_dir)
        result["status"] = "success"
        run_succeeded = True

    except Exception as exc:
        result.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise

    finally:
        cleanup_errors: list[str] = []

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                result["final_gpu_before_cleanup"] = gpu_stats()
            except Exception as exc:
                cleanup_errors.append(
                    f"Could not record final GPU stats: {exc}"
                )

        if model is not None:
            del model

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cleanup_errors.append(
                    f"CUDA cache cleanup failed: {exc}"
                )

        should_delete = (
            run_succeeded
            or bool(config.get("delete_temp_on_failure", True))
        )
        if should_delete:
            try:
                if run_root.exists():
                    shutil.rmtree(run_root)
                result["temporary_adapters_deleted"] = (
                    not run_root.exists()
                )
            except Exception as exc:
                result["temporary_adapters_deleted"] = False
                cleanup_errors.append(
                    f"Temporary adapter cleanup failed: {exc}"
                )
        else:
            result["temporary_adapters_deleted"] = False
            result["temporary_adapters_retained"] = str(run_root)

        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors

        result["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        write_result(result_path, result)


if __name__ == "__main__":
    main()
