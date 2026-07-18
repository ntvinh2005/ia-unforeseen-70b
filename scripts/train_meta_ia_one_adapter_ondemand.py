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
from collections import OrderedDict

import huggingface_hub
import peft
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download
from peft import PeftConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make "$PROJECT/..." paths in the JSON work even when PROJECT was not
# explicitly exported before starting Python.
os.environ.setdefault("PROJECT", str(PROJECT_ROOT))

# Override with IA_REPO_ROOT when the repository has a different name/location.
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
        f"Expected metalora.py at: {METALORA_FILE}\n"
        "Place this script in $PROJECT/scripts and the repository in "
        "$PROJECT/repos/introspection-adapters, or export IA_REPO_ROOT."
    )

# The repository root, not $PROJECT/repos, must be on sys.path so that
# `from src...` resolves to introspection-adapters/src.
sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.metalora import (  # noqa: E402
    convert_dataset_to_dataloader,
    train_meta_lora_preload,
)


DEFAULT_CONFIG_PATH = (
    SCRIPT_DIR
    / "configs"
    / "train_meta_ia_one_adapter_ondemand.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one AuditBench LoRA, train one meta-IA integration run, "
            "save the meta-IA, and delete the temporary behavior adapter."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Expand environment variables and resolve relative paths from $PROJECT."""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))

    if "$" in expanded:
        raise ValueError(
            f"Path contains an unresolved environment variable: {value}"
        )

    path = Path(expanded)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = {
        "huggingface_repo_id",
        "adapter_name",
        "expected_base_model_id",
        "base_model_path",
        "dataset_path",
        "temp_adapter_root",
        "output_dir",
    }
    missing = sorted(required_keys - config.keys())
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if config["adapter_name"] == "meta_lora":
        raise ValueError(
            "adapter_name cannot be 'meta_lora'; that name is reserved "
            "for the trainable introspection adapter."
        )

    if int(config.get("n_epochs", 1)) != 1:
        raise ValueError(
            "This integration script currently requires n_epochs=1 because "
            "the upstream checkpoint/resume logic is not correct for >1 epoch."
        )

    if int(config.get("k_adapters_per_step", 1)) != 1:
        raise ValueError(
            "This one-adapter integration experiment requires "
            "k_adapters_per_step=1."
        )

    for key in (
        "batch_size",
        "max_length",
        "max_samples",
        "save_steps",
        "meta_lora_r",
        "meta_lora_alpha",
    ):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be a positive integer")

    return config


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


def safe_name(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in value
    ).strip("._")
    return cleaned or "adapter"


def build_temp_adapter_dir(config: dict[str, Any]) -> Path:
    temp_root = resolve_path(config["temp_adapter_root"])
    run_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    adapter_dir = (
        temp_root
        / safe_name(config["adapter_name"])
        / safe_name(run_id)
    ).resolve()

    # We delete adapter_dir recursively in finally. Verify it is a child of
    # temp_root and is not temp_root itself.
    try:
        adapter_dir.relative_to(temp_root)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe temporary adapter path: {adapter_dir}"
        ) from exc

    if adapter_dir == temp_root:
        raise ValueError(
            f"Refusing to use temporary root itself as job directory: {temp_root}"
        )

    return adapter_dir


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
                "The output directory already contains checkpoints but "
                "allow_resume=false:\n"
                f"{output_dir}\n"
                "Use a new output directory or explicitly enable resume."
            )
        if final_adapter_dir.exists():
            raise RuntimeError(
                "The output directory already contains final_meta_lora but "
                "allow_resume=false:\n"
                f"{final_adapter_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, result_path


def resolve_model_and_dataset(
    config: dict[str, Any],
) -> tuple[Path, str]:
    base_model_path = resolve_path(config["base_model_path"])
    if not base_model_path.is_dir():
        raise FileNotFoundError(
            f"Local base model directory not found: {base_model_path}"
        )

    dataset_value = str(config["dataset_path"])
    use_hf_dataset = bool(config.get("use_hf_dataset", False))

    if dataset_value.startswith("hf://") or use_hf_dataset:
        dataset_source = dataset_value
    else:
        dataset_path = resolve_path(dataset_value)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Formatted prediction dataset not found: {dataset_path}"
            )
        dataset_source = str(dataset_path)

    return base_model_path, dataset_source


def download_adapter(
    config: dict[str, Any],
    adapter_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    repo_id = str(config["huggingface_repo_id"])
    requested_revision = config.get("revision")

    if adapter_dir.exists():
        if bool(config.get("clean_temp_before_download", True)):
            shutil.rmtree(adapter_dir)
        elif any(adapter_dir.iterdir()):
            raise RuntimeError(
                f"Temporary adapter directory is not empty: {adapter_dir}"
            )

    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Resolving revision for {repo_id} "
        f"(requested={requested_revision!r})...",
        flush=True,
    )
    resolve_start = time.perf_counter()
    repo_info = HfApi().model_info(
        repo_id=repo_id,
        revision=requested_revision,
    )
    resolved_revision = repo_info.sha
    revision_resolve_seconds = time.perf_counter() - resolve_start

    if not resolved_revision:
        raise RuntimeError(
            f"Could not resolve a commit SHA for {repo_id}"
        )

    print(
        f"Downloading adapter commit {resolved_revision} to {adapter_dir}...",
        flush=True,
    )
    download_start = time.perf_counter()

    try:
        returned_path = snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            allow_patterns=[
                "adapter_config.json",
                "adapter_model.safetensors",
            ],
            local_dir=str(adapter_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download AuditBench adapter: {repo_id}"
        ) from exc

    download_seconds = time.perf_counter() - download_start
    returned_dir = Path(returned_path).resolve()

    config_file = returned_dir / "adapter_config.json"
    weights_file = returned_dir / "adapter_model.safetensors"

    missing_files = [
        str(path)
        for path in (config_file, weights_file)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Downloaded adapter is incomplete. "
            f"Missing files: {missing_files}"
        )

    weights_size_bytes = weights_file.stat().st_size
    if weights_size_bytes <= 0:
        raise ValueError(
            f"Adapter weights file is empty: {weights_file}"
        )

    incomplete_files = list(returned_dir.rglob("*.incomplete"))
    if incomplete_files:
        raise RuntimeError(
            f"Incomplete download files remain: {incomplete_files}"
        )

    metadata = {
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "revision_resolve_seconds": revision_resolve_seconds,
        "download_seconds": download_seconds,
        "adapter_size_bytes": weights_size_bytes,
        "adapter_size_gib": weights_size_bytes / (1024**3),
        "adapter_local_path": str(returned_dir),
    }

    print(
        f"Download completed in {download_seconds:.2f}s; "
        f"adapter size={metadata['adapter_size_gib']:.3f} GiB",
        flush=True,
    )

    return returned_dir, metadata


def validate_adapter(
    adapter_dir: Path,
    expected_base_model_id: str,
    strict_base_model_match: bool,
) -> dict[str, Any]:
    peft_config = PeftConfig.from_pretrained(
        str(adapter_dir),
        local_files_only=True,
    )

    peft_type = getattr(
        peft_config.peft_type,
        "value",
        str(peft_config.peft_type),
    )
    task_type_object = getattr(peft_config, "task_type", None)
    task_type = getattr(
        task_type_object,
        "value",
        task_type_object,
    )
    declared_base = peft_config.base_model_name_or_path
    target_modules = getattr(peft_config, "target_modules", None)

    if peft_type != "LORA":
        raise ValueError(
            f"Expected PEFT type LORA, found {peft_type}"
        )

    if task_type not in (None, "CAUSAL_LM"):
        raise ValueError(
            f"Expected task type CAUSAL_LM, found {task_type}"
        )

    if not target_modules:
        raise ValueError("Adapter config has no target_modules")

    base_models_match = (
        str(declared_base).rstrip("/")
        == str(expected_base_model_id).rstrip("/")
    )
    if strict_base_model_match and not base_models_match:
        raise ValueError(
            "Adapter base-model mismatch:\n"
            f"  expected: {expected_base_model_id}\n"
            f"  declared: {declared_base}"
        )

    metadata = {
        "peft_type": peft_type,
        "task_type": task_type,
        "declared_base_model": declared_base,
        "expected_base_model": expected_base_model_id,
        "base_models_match": base_models_match,
        "target_modules": sorted(str(module) for module in target_modules),
        "rank": getattr(peft_config, "r", None),
        "lora_alpha": getattr(peft_config, "lora_alpha", None),
        "modules_to_save": getattr(peft_config, "modules_to_save", None),
        "inference_mode": getattr(peft_config, "inference_mode", None),
    }

    print("Adapter validation passed:", metadata, flush=True)
    return metadata


def format_function_for_formatted_dataset(
    item: dict[str, Any],
    _list_so_far: list[Any],
) -> list[list[dict[str, str]]]:
    user_prompt = item.get("prediction_user_prompt")
    assistant_response = item.get(
        "prediction_assistant_response"
    )

    if user_prompt is None or assistant_response is None:
        raise ValueError(
            "prediction_user_prompt or "
            "prediction_assistant_response is missing: "
            f"{item}"
        )

    messages: list[dict[str, str]] = []

    system_prompt = item.get("prediction_system_prompt")
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.extend(
        [
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ]
    )

    return [messages]


def inspect_dataloader(dataloader: Any) -> dict[str, Any]:
    try:
        first_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError("Dataloader contains no batches") from exc

    required_fields = {"input_ids", "attention_mask", "labels"}
    missing_fields = required_fields - first_batch.keys()
    if missing_fields:
        raise KeyError(
            f"First batch is missing fields: {sorted(missing_fields)}"
        )

    input_ids = first_batch["input_ids"]
    attention_mask = first_batch["attention_mask"]
    labels = first_batch["labels"]

    if (
        input_ids.shape != attention_mask.shape
        or input_ids.shape != labels.shape
    ):
        raise ValueError(
            "input_ids, attention_mask, and labels must have "
            f"matching shapes; found {input_ids.shape}, "
            f"{attention_mask.shape}, and {labels.shape}"
        )

    supervised_tokens = int((labels != -100).sum().item())
    if supervised_tokens <= 0:
        raise ValueError(
            "The first batch has no supervised assistant tokens"
        )

    return {
        "number_of_batches": len(dataloader),
        "first_batch_shape": list(input_ids.shape),
        "first_batch_supervised_tokens": supervised_tokens,
        "input_ids_dtype": str(input_ids.dtype),
        "labels_dtype": str(labels.dtype),
    }


def read_loss_summary(output_dir: Path) -> dict[str, Any]:
    loss_path = output_dir / "loss_curve.json"
    if not loss_path.is_file():
        return {
            "loss_curve_path": str(loss_path),
            "loss_curve_found": False,
        }

    with loss_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    summary: dict[str, Any] = {
        "loss_curve_path": str(loss_path),
        "loss_curve_found": True,
        "loss_entries": len(history),
    }

    if history:
        summary.update(
            {
                "first_recorded_step": history[0].get("step"),
                "first_recorded_loss": history[0].get("loss"),
                "last_recorded_step": history[-1].get("step"),
                "last_recorded_loss": history[-1].get("loss"),
            }
        )

    return summary


def write_result(result_path: Path, result: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Result written to {result_path}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    base_model_path, dataset_source = resolve_model_and_dataset(config)
    output_dir, result_path = prepare_output_paths(config)
    temp_adapter_dir = build_temp_adapter_dir(config)

    adapter_name = str(config["adapter_name"])
    result: dict[str, Any] = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "config_path": str(resolve_path(args.config)),
        "adapter_name": adapter_name,
        "adapter_repo": config["huggingface_repo_id"],
        "requested_adapter_revision": config.get("revision"),
        "base_model_path": str(base_model_path),
        "dataset_source": dataset_source,
        "output_dir": str(output_dir),
        "temporary_adapter_dir": str(temp_adapter_dir),
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
    }

    model = None
    downloaded_adapter_dir: Path | None = None
    run_succeeded = False

    try:
        downloaded_adapter_dir, download_metadata = download_adapter(
            config,
            temp_adapter_dir,
        )
        result["download"] = download_metadata

        result["adapter_config"] = validate_adapter(
            downloaded_adapter_dir,
            expected_base_model_id=str(
                config["expected_base_model_id"]
            ),
            strict_base_model_match=bool(
                config.get("strict_base_model_match", True)
            ),
        )

        print("Preparing formatted prediction dataloader...", flush=True)
        dataloader_start = time.perf_counter()
        dataloader = convert_dataset_to_dataloader(
            dataset_id_or_path=dataset_source,
            use_HF=bool(config.get("use_hf_dataset", False)),
            tokenizer_name=str(base_model_path),
            batch_size=int(config.get("batch_size", 1)),
            max_length=int(config.get("max_length", 256)),
            max_samples=int(config.get("max_samples", 1)),
            seed=int(config.get("seed", 42)),
            format_function=format_function_for_formatted_dataset,
        )
        result["dataloader"] = inspect_dataloader(dataloader)
        result["dataloader"]["preparation_seconds"] = (
            time.perf_counter() - dataloader_start
        )

        # The adapter and dataloader dictionaries must use the same key.
        lora_adapters = {
            adapter_name: str(downloaded_adapter_dir),
        }
        dataloaders = {
            adapter_name: dataloader,
        }

        print(
            "Starting one-adapter meta-IA training...",
            flush=True,
        )
        training_start = time.perf_counter()

        model = train_meta_lora_preload(
            base_model_name=str(base_model_path),
            lora_adapters=lora_adapters,
            dataloaders=dataloaders,
            learning_rate=float(
                config.get("learning_rate", 1e-4)
            ),
            n_epochs=int(config.get("n_epochs", 1)),
            seed=int(config.get("seed", 42)),
            k_adapters_per_step=int(
                config.get("k_adapters_per_step", 1)
            ),
            r=int(config.get("meta_lora_r", 16)),
            lora_alpha=int(
                config.get("meta_lora_alpha", 32)
            ),
            meta_lora_init_path=(
                None
                if config.get("meta_lora_init_path") is None
                else str(
                    resolve_path(
                        config["meta_lora_init_path"]
                    )
                )
            ),
            output_dir=str(output_dir),
            save_steps=int(config.get("save_steps", 1)),
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

        # Record memory even for failed/OOM runs when CUDA still responds.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                result["final_gpu_before_cleanup"] = gpu_stats()
            except Exception as exc:
                cleanup_errors.append(
                    f"Could not record final GPU stats: {exc}"
                )

        if model is not None:
            try:
                model.set_adapter("meta_lora")
                if (
                    hasattr(model, "peft_config")
                    and adapter_name in model.peft_config
                ):
                    model.delete_adapter(adapter_name)
                    print(
                        f"Deleted behavior adapter '{adapter_name}' "
                        "from the PEFT model",
                        flush=True,
                    )
            except Exception as exc:
                cleanup_errors.append(
                    f"PEFT adapter cleanup failed: {exc}"
                )

            del model

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                cleanup_errors.append(
                    f"CUDA cache cleanup failed: {exc}"
                )

        should_delete_temp = (
            run_succeeded
            or bool(config.get("delete_temp_on_failure", True))
        )

        if should_delete_temp:
            try:
                if temp_adapter_dir.exists():
                    shutil.rmtree(temp_adapter_dir)

                result["temporary_adapter_deleted"] = (
                    not temp_adapter_dir.exists()
                )
                print(
                    "Temporary adapter directory deleted: "
                    f"{temp_adapter_dir}",
                    flush=True,
                )
            except Exception as exc:
                result["temporary_adapter_deleted"] = False
                cleanup_errors.append(
                    f"Temporary disk cleanup failed: {exc}"
                )
        else:
            result["temporary_adapter_deleted"] = False
            result["temporary_adapter_retained_for_debugging"] = str(
                temp_adapter_dir
            )

        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors

        result["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        write_result(result_path, result)


if __name__ == "__main__":
    main()