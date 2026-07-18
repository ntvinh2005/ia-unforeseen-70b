# To run, run this in bash
# python scripts/eval_meta_ia_one_adapter.py \
#     --config scripts/configs/train_meta_ia_one_adapter_ondemand.json \
#     --num-prompts 3

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT", SCRIPT_DIR.parent)).expanduser().resolve()
REPO_ROOT = Path(
    os.environ.get(
        "IA_REPO_ROOT",
        PROJECT_ROOT / "repos" / "introspection-adapters",
    )
).expanduser().resolve()

if not (REPO_ROOT / "src" / "utils" / "lora_evals.py").is_file():
    raise RuntimeError(
        "Cannot find introspection-adapters repo. Expected: "
        f"{REPO_ROOT / 'src/utils/lora_evals.py'}\n"
        "Set IA_REPO_ROOT or place the repo at "
        "$PROJECT/repos/introspection-adapters."
    )

sys.path.insert(0, str(REPO_ROOT))
from src.utils.lora_evals import eval_over_many_adapters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a one-example meta-IA with sanity-check controls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of prompts sampled from the formatted prediction dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overrides config seed.",
    )
    parser.add_argument(
        "--skip-base-controls",
        action="store_true",
        help="Skip IA-on-base and raw-base generations to reduce runtime.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = load_json(config_path)
    required = {
        "huggingface_repo_id",
        "adapter_name",
        "base_model_path",
        "dataset_path",
        "output_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")
    return config


def download_behavior_adapter(config: dict[str, Any]) -> tuple[Path, str]:
    repo_id = str(config["huggingface_repo_id"])
    requested_revision = config.get("revision")
    info = HfApi().model_info(repo_id=repo_id, revision=requested_revision)
    revision = info.sha
    if not revision:
        raise RuntimeError(f"Could not resolve commit SHA for {repo_id}")

    temp_root = resolve_path(
        config.get("eval_temp_adapter_root", "$PROJECT/tmp/eval_one_adapter")
    )
    run_name = f"{config['adapter_name']}-{os.getpid()}"
    adapter_dir = temp_root / run_name
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    returned = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
        local_dir=str(adapter_dir),
    )
    adapter_path = Path(returned).resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / filename).is_file():
            raise FileNotFoundError(f"Missing downloaded file: {adapter_path / filename}")
    return adapter_path, revision


def load_formatted_prompts(
    dataset_source: str,
    num_prompts: int,
    seed: int,
    use_hf_dataset: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    import random

    if dataset_source.startswith("hf://") or use_hf_dataset:
        from datasets import load_dataset

        dataset_spec = dataset_source.removeprefix("hf://")
        parts = dataset_spec.split("/")

        dataset_id = "/".join(parts[:2])
        data_dir = "/".join(parts[2:]) or None

        print(
            f"Loading Hugging Face dataset: "
            f"id={dataset_id}, data_dir={data_dir}",
            flush=True,
        )

        loaded = load_dataset(
            dataset_id,
            data_dir=data_dir,
        )

        if hasattr(loaded, "keys"):
            if "train" in loaded:
                dataset = loaded["train"]
            else:
                first_split = next(iter(loaded.keys()))
                print(
                    f"No train split found; using split '{first_split}'",
                    flush=True,
                )
                dataset = loaded[first_split]
        else:
            dataset = loaded

        rows = [dict(row) for row in dataset]
    else:
        dataset_path = resolve_path(dataset_source)

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Local evaluation dataset not found: {dataset_path}"
            )

        with dataset_path.open("r", encoding="utf-8") as file:
            if dataset_path.suffix == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]
            else:
                loaded = json.load(file)

                if isinstance(loaded, list):
                    rows = loaded
                elif isinstance(loaded, dict):
                    for key in ("train", "data", "examples"):
                        if key in loaded and isinstance(loaded[key], list):
                            rows = loaded[key]
                            break
                    else:
                        raise ValueError(
                            "JSON object does not contain a supported "
                            "list field such as train, data, or examples."
                        )
                else:
                    raise ValueError(
                        "Expected a JSON list or JSON object containing rows."
                    )

    valid_rows = []

    for row in rows:
        user_prompt = row.get("prediction_user_prompt")
        target = row.get("prediction_assistant_response")

        if user_prompt is None or target is None:
            continue

        valid_rows.append(row)

    if not valid_rows:
        available_columns = (
            sorted(rows[0].keys()) if rows else []
        )
        raise ValueError(
            "No rows contain both prediction_user_prompt and "
            "prediction_assistant_response. "
            f"Available columns: {available_columns}"
        )

    rng = random.Random(seed)
    rng.shuffle(valid_rows)
    selected_rows = valid_rows[:num_prompts]

    prompts: list[list[dict[str, str]]] = []

    for row in selected_rows:
        messages: list[dict[str, str]] = []

        system_prompt = row.get("prediction_system_prompt")
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": str(system_prompt),
                }
            )

        user_prompt = row.get("prediction_user_prompt")
        if not user_prompt:
            raise ValueError(
                "Selected dataset row is missing prediction_user_prompt: "
                f"{row}"
            )

        messages.append(
            {
                "role": "user",
                "content": str(user_prompt),
            }
        )

        prompts.append(messages)

    # Fail early before loading the 70B model if the format is wrong.
    for index, messages in enumerate(prompts):
        if not isinstance(messages, list):
            raise TypeError(
                f"Prompt {index} must be a list of chat messages, "
                f"found {type(messages).__name__}"
            )

        for message in messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                raise TypeError(
                    f"Invalid message in prompt {index}: {message!r}"
                )

    print("First formatted prompt:", prompts[0], flush=True)

    return prompts, selected_rows


def normalize_official_results(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(f"Unexpected eval result container: {type(raw)}")
    result_list = raw[0]
    if not isinstance(result_list, list):
        raise ValueError(f"Unexpected per-adapter result: {type(result_list)}")
    return result_list


def infer_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def generate_direct(
    *,
    base_model_path: Path,
    ia_path: Path | None,
    prompts: list[list[dict[str, str]]],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        local_files_only=True,
        torch_dtype=infer_dtype(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = base
    if ia_path is not None:
        model = PeftModel.from_pretrained(
            base,
            str(ia_path),
            adapter_name="meta_lora",
            is_trainable=False,
            local_files_only=True,
        )
        model.set_adapter("meta_lora")
    model.eval()

    outputs: list[str] = []
    with torch.inference_mode():
        for messages in prompts:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            ).to(model.device)
            attention_mask = torch.ones_like(encoded)
            generated = model.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[0, encoded.shape[1] :]
            outputs.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())

    del model
    if ia_path is not None:
        del base
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    base_model_path = resolve_path(config["base_model_path"])
    ia_path = resolve_path(
        config.get(
            "meta_lora_eval_path",
            resolve_path(config["output_dir"]) / "final_meta_lora",
        )
    )
    if not ia_path.is_dir():
        raise FileNotFoundError(f"Trained IA not found: {ia_path}")

    prompts, selected_rows = load_formatted_prompts(
        dataset_source=str(config["dataset_path"]),
        num_prompts=args.num_prompts,
        seed=seed,
        use_hf_dataset=bool(config.get("use_hf_dataset", False)),
    )
    behavior_adapter_path, resolved_revision = download_behavior_adapter(config)

    eval_config = {
        "batch_size": int(config.get("eval_batch_size", 1)),
        "max_input_tokens": int(config.get("eval_max_input_tokens", 2048)),
        "max_new_tokens": int(config.get("eval_max_new_tokens", 128)),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
    }

    started = time.perf_counter()
    official_raw = eval_over_many_adapters(
        tokenizer_name=str(base_model_path),
        base_model=str(base_model_path),
        meta_lora_path=str(ia_path),
        adapter_and_datasets=[
            {
                "adapter_name": str(config["adapter_name"]),
                "finetune_dataset": prompts,
                "baseline_dataset": prompts,
                "adapter_path": str(behavior_adapter_path),
            }
        ],
        config=eval_config,
        # False gives both behavior+IA and behavior-only outputs.
        skip_baseline=False,
    )
    official_results = normalize_official_results(official_raw)

    base_plus_ia: list[str] | None = None
    raw_base: list[str] | None = None
    if not args.skip_base_controls:
        gc.collect()
        torch.cuda.empty_cache()
        base_plus_ia = generate_direct(
            base_model_path=base_model_path,
            ia_path=ia_path,
            prompts=prompts,
            max_input_tokens=eval_config["max_input_tokens"],
            max_new_tokens=eval_config["max_new_tokens"],
        )
        raw_base = generate_direct(
            base_model_path=base_model_path,
            ia_path=None,
            prompts=prompts,
            max_input_tokens=eval_config["max_input_tokens"],
            max_new_tokens=eval_config["max_new_tokens"],
        )

    rows_out: list[dict[str, Any]] = []
    for index, source_row in enumerate(selected_rows):
        official = official_results[index]
        rows_out.append(
            {
                "index": index,
                "prediction_system_prompt": source_row.get("prediction_system_prompt"),
                "prediction_user_prompt": source_row["prediction_user_prompt"],
                "training_target": source_row.get("prediction_assistant_response"),
                "behavior_plus_ia": official.get("meta_lora_response"),
                "behavior_without_ia": (
                    official.get("baseline_response")
                    or official.get("finetune_response")
                    or official.get("adapter_response")
                ),
                "base_plus_ia": None if base_plus_ia is None else base_plus_ia[index],
                "raw_base": None if raw_base is None else raw_base[index],
                "official_result_raw": official,
            }
        )

    output_dir = resolve_path(config["output_dir"])
    output_path = resolve_path(
        config.get(
            "one_adapter_eval_result_path",
            output_dir / "one_adapter_sanity_eval.json",
        )
    )
    result = {
        "status": "success",
        "base_model_path": str(base_model_path),
        "meta_lora_path": str(ia_path),
        "behavior_adapter_repo": config["huggingface_repo_id"],
        "behavior_adapter_revision": resolved_revision,
        "behavior_adapter_local_path": str(behavior_adapter_path),
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "conditions": {
            "behavior_plus_ia": "Base + behavior LoRA + trained meta-IA",
            "behavior_without_ia": "Base + behavior LoRA, without IA",
            "base_plus_ia": "Base + trained meta-IA, without behavior LoRA",
            "raw_base": "Base model only",
        },
        "rows": rows_out,
    }
    save_json(output_path, result)

    print(f"Saved evaluation to: {output_path}")
    print("\n" + "=" * 80)
    for row in rows_out:
        print(f"PROMPT: {row['prediction_user_prompt']}")
        print(f"TARGET: {row['training_target']}")
        print(f"BEHAVIOR + IA: {row['behavior_plus_ia']}")
        print(f"BEHAVIOR ONLY: {row['behavior_without_ia']}")
        if not args.skip_base_controls:
            print(f"BASE + IA: {row['base_plus_ia']}")
            print(f"RAW BASE: {row['raw_base']}")
        print("-" * 80)

    if bool(config.get("delete_eval_temp_adapter", True)):
        shutil.rmtree(behavior_adapter_path, ignore_errors=True)


if __name__ == "__main__":
    main()
