from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("PROJECT", str(PROJECT_ROOT))

REPO_ROOT = Path(
    os.environ.get(
        "IA_REPO_ROOT",
        str(PROJECT_ROOT / "repos" / "introspection-adapters"),
    )
).expanduser().resolve()

UTILS_FILE = REPO_ROOT / "src" / "utils" / "utils.py"
if not UTILS_FILE.is_file():
    raise RuntimeError(
        "Cannot find introspection-adapters repository utilities.\n"
        f"Expected: {UTILS_FILE}\n"
        "Set IA_REPO_ROOT or place the repository under "
        "$PROJECT/repos/introspection-adapters."
    )

sys.path.insert(0, str(REPO_ROOT))
from src.utils.utils import load_prediction_dataset  # noqa: E402


ASSISTANT_START_PLACEHOLDER = "<|ASSISTANT_START_MARKER|>"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "train_meta_ia_multi_adapter.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained shared Meta-IA over behavior adapters with "
            "deterministic generation, teacher-forced target NLL, and base controls."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config JSON.",
    )
    parser.add_argument(
        "--meta-lora-path",
        type=str,
        default=None,
        help="Override path to trained meta adapter. Default: <config.output_dir>/final_meta_lora",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: <config.output_dir>/eval_<prompt_split>.json",
    )
    parser.add_argument(
        "--adapter-limit",
        type=int,
        default=None,
        help="Optional limit on number of adapters (after manifest filtering).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Prompts sampled per adapter from the requested split. Default: all prompts in split.",
    )
    parser.add_argument(
        "--prompt-split",
        choices=("seen", "unseen", "all"),
        default="seen",
        help="Prompt split relative to training-row sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed for evaluation prompt subsampling. Default: config.seed",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum newly generated tokens for deterministic generation.",
    )
    parser.add_argument(
        "--skip-base-controls",
        action="store_true",
        help="Skip base_plus_ia and base_only controls to reduce runtime.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in expanded:
        raise ValueError(f"Path contains an unresolved environment variable: {value}")
    path = Path(expanded)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_name(value: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value
    ).strip("._")
    return cleaned or "adapter"


def load_json(path_value: str | Path) -> Any:
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2, ensure_ascii=False)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def load_training_config(path_value: str | Path) -> dict[str, Any]:
    config = load_json(path_value)
    required = {
        "adapter_manifest_path",
        "base_model_path",
        "output_dir",
        "max_samples_per_adapter",
        "seed",
        "max_length",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")
    return config


def load_manifest(path_value: str | Path, adapter_limit: int | None) -> list[dict[str, Any]]:
    raw = load_json(path_value)
    entries = raw.get("adapters") if isinstance(raw, dict) else raw
    manifest_dataset_revision = (
        raw.get("dataset_revision") if isinstance(raw, dict) else None
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must be a non-empty list or an object with non-empty 'adapters'"
        )

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for i, item in enumerate(entries):
        if not isinstance(item, dict):
            raise TypeError(f"Manifest entry {i} is not an object")
        if item.get("enabled", True) is False:
            continue

        repo_id = item.get("repo_id") or item.get("huggingface_repo_id")
        dataset_path = item.get("dataset_path")
        adapter_name = item.get("adapter_name") or item.get("name")

        if not repo_id or not dataset_path:
            raise KeyError(f"Manifest entry {i} requires repo_id and dataset_path")

        if not adapter_name:
            adapter_name = safe_name(str(repo_id).split("/")[-1])

        adapter_name = safe_name(str(adapter_name))
        if adapter_name == "meta_lora":
            raise ValueError("Manifest adapter_name 'meta_lora' is reserved")
        if adapter_name in seen_names:
            raise ValueError(f"Duplicate adapter_name in manifest: {adapter_name}")
        seen_names.add(adapter_name)

        revision = item.get("revision")
        if not revision:
            raise ValueError(
                f"Manifest entry '{adapter_name}' is missing pinned revision; this evaluator requires pinned revisions."
            )
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in revision
        ):
            raise ValueError(
                f"Manifest entry '{adapter_name}' revision must be a 40-character commit SHA"
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
            if (
                not isinstance(dataset_revision, str)
                or len(dataset_revision) != 40
                or any(
                    character not in "0123456789abcdefABCDEF"
                    for character in dataset_revision
                )
            ):
                raise ValueError(
                    f"Manifest entry '{adapter_name}' dataset_revision must "
                    "be a 40-character commit SHA"
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
                    f"Manifest entry '{adapter_name}' dataset_sha256 must "
                    "be a 64-character SHA-256 digest"
                )
            if (
                not isinstance(dataset_size_bytes, int)
                or isinstance(dataset_size_bytes, bool)
                or dataset_size_bytes < 0
            ):
                raise ValueError(
                    f"Manifest entry '{adapter_name}' dataset_size_bytes "
                    "must be a non-negative integer"
                )
            hf_parts = dataset_path_string[len("hf://"):].split("/", 2)
            if len(hf_parts) != 3 or not all(hf_parts):
                raise ValueError(
                    f"Manifest entry '{adapter_name}' has invalid hf:// dataset_path"
                )
            expected_dataset_file = f"{hf_parts[2]}/eval.jsonl"
            if dataset_file != expected_dataset_file:
                raise ValueError(
                    f"Manifest entry '{adapter_name}' dataset_file must be "
                    f"{expected_dataset_file!r}"
                )

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
                "use_hf_dataset": bool(item.get("use_hf_dataset", False)),
                "max_samples": item.get("max_samples"),
            }
        )

    if adapter_limit is not None:
        if adapter_limit <= 0:
            raise ValueError("--adapter-limit must be positive")
        normalized = normalized[:adapter_limit]

    if not normalized:
        raise ValueError("No enabled adapters remain after filtering")

    return normalized


def resolve_dataset_source(dataset_path: str) -> str:
    if dataset_path.startswith("hf://"):
        return dataset_path
    return str(resolve_path(dataset_path))


def build_run_adapter_root(config: dict[str, Any]) -> Path:
    root_value = config.get("temp_adapter_root") or "$SLURM_TMPDIR/meta_ia_eval_adapters"
    root = resolve_path(root_value)
    run_id = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    run_root = (root / f"meta-ia-eval-{safe_name(run_id)}").resolve()

    try:
        run_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe adapter run directory: {run_root}") from exc

    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def list_word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize_text(text: str) -> str:
    return " ".join(list_word_tokens(text))


def token_precision_recall_f1(prediction: str, target: str) -> tuple[float, float, float]:
    pred_tokens = list_word_tokens(prediction)
    target_tokens = list_word_tokens(target)

    if not pred_tokens and not target_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    if not target_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    target_counts = Counter(target_tokens)
    overlap = sum(min(pred_counts[tok], target_counts[tok]) for tok in pred_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def apply_chat_template_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_length,
    }
    try:
        kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    user_prompt = row.get("prediction_user_prompt")
    target = row.get("prediction_assistant_response")
    if user_prompt is None or target is None:
        raise ValueError(
            "Dataset row is missing prediction_user_prompt or prediction_assistant_response"
        )

    messages: list[dict[str, str]] = []
    system_prompt = row.get("prediction_system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(user_prompt)})
    return messages


def build_supervised_tensors(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    target_text: str,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    messages_with_placeholder = list(prompt_messages)
    messages_with_placeholder.append(
        {
            "role": "assistant",
            "content": ASSISTANT_START_PLACEHOLDER + target_text,
        }
    )

    full_text = apply_chat_template_text(
        tokenizer,
        messages_with_placeholder,
        add_generation_prompt=False,
    )
    parts = full_text.split(ASSISTANT_START_PLACEHOLDER)
    if len(parts) != 2:
        raise ValueError("Assistant-start placeholder split failed while building labels")

    prompt_text = parts[0]
    assistant_text = parts[1]

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    assistant_tokens = tokenizer(assistant_text, add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + assistant_tokens["input_ids"]
    attention_mask = prompt_tokens["attention_mask"] + assistant_tokens["attention_mask"]
    labels = ([-100] * len(prompt_tokens["input_ids"])) + assistant_tokens["input_ids"]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    supervised_tokens = sum(1 for value in labels if value != -100)
    if supervised_tokens <= 0:
        raise ValueError(
            "No supervised assistant tokens remain after truncation; increase max_length or inspect dataset row formatting."
        )

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_tensor = torch.tensor([attention_mask], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    return input_tensor, attention_tensor, labels_tensor, supervised_tokens


@contextmanager
def active_condition(
    model: PeftModel,
    condition: str,
    behavior_adapter_name: str,
):
    """Activate the exact adapter combination required for one eval condition.

    PeftModel.set_adapter() in the installed PEFT version accepts only one
    adapter name. For multiple LoRA adapters, PEFT requires calling
    model.base_model.set_adapter([...]).
    """
    available_adapters = set(model.peft_config.keys())

    if condition == "base_only":
        # disable_adapter() temporarily disables every PEFT adapter while
        # preserving the loaded adapter objects.
        try:
            with model.disable_adapter():
                freeze_all_params(model)
                yield
        finally:
            # Always return to a known state for the next condition.
            model.set_adapter("meta_lora")
            freeze_all_params(model)
        return

    if condition == "behavior_plus_ia":
        requested_adapters = [
            behavior_adapter_name,
            "meta_lora",
        ]
    elif condition == "behavior_only":
        requested_adapters = [
            behavior_adapter_name,
        ]
    elif condition == "base_plus_ia":
        requested_adapters = [
            "meta_lora",
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    missing_adapters = [
        adapter_name
        for adapter_name in requested_adapters
        if adapter_name not in available_adapters
    ]
    if missing_adapters:
        raise RuntimeError(
            "Requested adapter condition cannot be activated.\n"
            f"Condition: {condition}\n"
            f"Requested: {requested_adapters}\n"
            f"Missing: {missing_adapters}\n"
            f"Available: {sorted(available_adapters)}"
        )

    if len(requested_adapters) == 1:
        # The outer PeftModel API accepts one adapter name.
        model.set_adapter(requested_adapters[0])
    else:
        # The BaseTuner API supports simultaneous LoRA activation.
        model.base_model.set_adapter(requested_adapters)

    # set_adapter() may mark active adapters trainable. Evaluation must keep
    # all parameters frozen.
    freeze_all_params(model)

    actual_active = getattr(model.base_model, "active_adapters", None)
    if actual_active is not None:
        if isinstance(actual_active, str):
            actual_active_list = [actual_active]
        else:
            actual_active_list = list(actual_active)

        if (
            len(actual_active_list) != len(requested_adapters)
            or set(actual_active_list) != set(requested_adapters)
        ):
            raise RuntimeError(
                "PEFT did not activate the requested adapter combination.\n"
                f"Condition: {condition}\n"
                f"Requested: {requested_adapters}\n"
                f"Actually active: {actual_active_list}"
            )

        print(
            f"PEFT active adapters for {condition}: {actual_active_list}",
            flush=True,
        )

    try:
        yield
    finally:
        # Prevent adapter state from leaking into the following condition.
        model.set_adapter("meta_lora")
        freeze_all_params(model)

def generate_completion(
    *,
    model: PeftModel,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    encoded = apply_chat_template_tokens(
        tokenizer=tokenizer,
        messages=prompt_messages,
        max_length=max_input_tokens,
    ).to(device)
    attention_mask = torch.ones_like(encoded)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=encoded,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_only = generated[0, encoded.shape[1] :]
    return tokenizer.decode(generated_only, skip_special_tokens=True).strip()


def compute_target_nll(
    *,
    model: PeftModel,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    target_text: str,
    max_input_tokens: int,
    device: torch.device,
) -> tuple[float, int]:
    input_ids, attention_mask, labels, supervised_tokens = build_supervised_tensors(
        tokenizer=tokenizer,
        prompt_messages=prompt_messages,
        target_text=target_text,
        max_length=max_input_tokens,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid_mask = shift_labels != -100
    if int(valid_mask.sum().item()) <= 0:
        raise ValueError("No supervised target tokens after shift for cross-entropy")

    loss_per_token = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)

    mean_nll = float(loss_per_token[valid_mask].mean().item())
    return mean_nll, supervised_tokens


def summarize_condition(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [r for r in records if r["condition"] == condition]
    if not subset:
        return {
            "count": 0,
            "mean_target_nll": 0.0,
            "mean_token_f1": 0.0,
            "exact_match_rate": 0.0,
        }

    return {
        "count": len(subset),
        "mean_target_nll": float(sum(r["target_nll"] for r in subset) / len(subset)),
        "mean_token_f1": float(sum(r["token_f1"] for r in subset) / len(subset)),
        "exact_match_rate": float(sum(1.0 if r["exact_match"] else 0.0 for r in subset) / len(subset)),
    }


def compute_gain_stats(records: list[dict[str, Any]]) -> tuple[list[float], dict[tuple[str, int], float]]:
    behavior_only: dict[tuple[str, int], float] = {}
    behavior_plus_ia: dict[tuple[str, int], float] = {}

    for record in records:
        key = (record["adapter_name"], int(record["dataset_index"]))
        if record["condition"] == "behavior_only":
            behavior_only[key] = float(record["target_nll"])
        elif record["condition"] == "behavior_plus_ia":
            behavior_plus_ia[key] = float(record["target_nll"])

    gains: list[float] = []
    gains_by_key: dict[tuple[str, int], float] = {}
    for key, behavior_nll in behavior_only.items():
        if key in behavior_plus_ia:
            gain = behavior_nll - behavior_plus_ia[key]
            gains.append(gain)
            gains_by_key[key] = gain

    return gains, gains_by_key


def recompute_summaries(
    *,
    payload: dict[str, Any],
    selected_conditions: list[str],
) -> None:
    records: list[dict[str, Any]] = payload["records"]

    summary: dict[str, Any] = {}
    for condition in selected_conditions:
        summary[condition] = summarize_condition(records, condition)

    gains, gains_by_key = compute_gain_stats(records)
    if gains:
        mean_gain = float(sum(gains) / len(gains))
        median_gain = float(statistics.median(gains))
        fraction_positive = float(sum(1.0 for g in gains if g > 0.0) / len(gains))
    else:
        mean_gain = 0.0
        median_gain = 0.0
        fraction_positive = 0.0

    summary["mean_ia_target_nll_gain_over_behavior_only"] = mean_gain
    summary["median_ia_target_nll_gain_over_behavior_only"] = median_gain
    summary["fraction_positive_ia_gain"] = fraction_positive

    behavior_plus = summary.get("behavior_plus_ia", {"mean_target_nll": 0.0, "count": 0})
    behavior_only = summary.get("behavior_only", {"mean_target_nll": 0.0, "count": 0})
    summary["ia_improves_mean_target_nll_vs_behavior_only"] = bool(
        behavior_plus.get("count", 0) > 0
        and behavior_only.get("count", 0) > 0
        and behavior_plus.get("mean_target_nll", 0.0) < behavior_only.get("mean_target_nll", 0.0)
    )

    if "base_plus_ia" in summary:
        summary["base_plus_ia_exact_match_rate"] = float(summary["base_plus_ia"]["exact_match_rate"])
        summary["base_plus_ia_memorization_flag"] = bool(summary["base_plus_ia"]["exact_match_rate"] > 0.0)

    payload["summary"] = summary

    per_adapter: dict[str, Any] = {}
    adapter_names = sorted({str(r["adapter_name"]) for r in records})
    for adapter_name in adapter_names:
        adapter_records = [r for r in records if r["adapter_name"] == adapter_name]
        adapter_summary: dict[str, Any] = {
            condition: summarize_condition(adapter_records, condition)
            for condition in selected_conditions
        }

        adapter_gains = [
            gains_by_key[(adapter_name, int(r["dataset_index"]))]
            for r in adapter_records
            if r["condition"] == "behavior_plus_ia"
            and (adapter_name, int(r["dataset_index"])) in gains_by_key
        ]

        if adapter_gains:
            adapter_summary["mean_ia_target_nll_gain_over_behavior_only"] = float(
                sum(adapter_gains) / len(adapter_gains)
            )
            adapter_summary["median_ia_target_nll_gain_over_behavior_only"] = float(
                statistics.median(adapter_gains)
            )
            adapter_summary["fraction_positive_ia_gain"] = float(
                sum(1.0 for g in adapter_gains if g > 0.0) / len(adapter_gains)
            )
        else:
            adapter_summary["mean_ia_target_nll_gain_over_behavior_only"] = 0.0
            adapter_summary["median_ia_target_nll_gain_over_behavior_only"] = 0.0
            adapter_summary["fraction_positive_ia_gain"] = 0.0

        per_adapter[adapter_name] = adapter_summary

    payload["per_adapter_summary"] = per_adapter


def cleanup_adapter_scratch(run_root: Path, adapter_name: str) -> None:
    adapter_dir = run_root / safe_name(adapter_name)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    for child in run_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def freeze_all_params(model: PeftModel) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def assert_no_meta_params_for_adapter(model: PeftModel, adapter_name: str) -> None:
    meta_param_names = [
        name
        for name, param in model.named_parameters()
        if adapter_name in name and getattr(param, "is_meta", False)
    ]
    if meta_param_names:
        raise RuntimeError(
            "Found meta-device parameters in loaded adapter "
            f"'{adapter_name}': {meta_param_names[:5]}"
        )


def ensure_expected_row_schema(rows: list[dict[str, Any]], dataset_source: str) -> None:
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Dataset row {idx} from {dataset_source} is not an object")
        if "prediction_user_prompt" not in row or "prediction_assistant_response" not in row:
            raise ValueError(
                "Dataset row is missing required fields "
                "prediction_user_prompt/prediction_assistant_response at index "
                f"{idx} in {dataset_source}"
            )


def select_indices_for_split(
    *,
    num_rows: int,
    prompt_split: str,
    train_seed: int,
    train_max_samples: int,
) -> list[int]:
    if train_max_samples > num_rows:
        raise ValueError(
            "Training max_samples_per_adapter exceeds dataset length for split reconstruction: "
            f"max_samples={train_max_samples}, rows={num_rows}"
        )

    rng = random.Random(train_seed)
    seen_indices = rng.sample(range(num_rows), train_max_samples)

    if prompt_split == "seen":
        return list(seen_indices)

    if prompt_split == "unseen":
        seen_set = set(seen_indices)
        unseen = [idx for idx in range(num_rows) if idx not in seen_set]
        if not unseen:
            raise RuntimeError(
                "No unseen prompt rows exist for this adapter because training consumed all rows. "
                "Do not treat seen rows as unseen."
            )
        return unseen

    if prompt_split == "all":
        return list(range(num_rows))

    raise ValueError(f"Unsupported prompt split: {prompt_split}")


def choose_prompt_indices(
    split_indices: list[int],
    num_prompts: int | None,
    seed: int,
    adapter_position: int,
) -> list[int]:
    if num_prompts is None:
        return list(split_indices)

    if num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")

    if num_prompts > len(split_indices):
        raise ValueError(
            f"Requested num-prompts={num_prompts} but split only contains {len(split_indices)} rows"
        )

    rng = random.Random(seed + (adapter_position * 1_000_003))
    return rng.sample(split_indices, num_prompts)


def load_behavior_rows(
    dataset_path: str,
    *,
    dataset_revision: str | None,
    dataset_sha256: str | None,
    dataset_size_bytes: int | None,
) -> list[dict[str, Any]]:
    dataset_source = resolve_dataset_source(dataset_path)
    rows = load_prediction_dataset(
        dataset_source,
        revision=dataset_revision,
        expected_sha256=dataset_sha256,
        expected_size_bytes=dataset_size_bytes,
    )
    if not isinstance(rows, list):
        raise TypeError(f"Prediction dataset is not a list: {dataset_source}")
    ensure_expected_row_schema(rows, dataset_source)
    return rows


def make_initial_payload(
    *,
    base_model_path: Path,
    meta_lora_path: Path,
    manifest_path: Path,
    training_config_path: Path,
    prompt_split: str,
    conditions: list[str],
    number_of_adapters: int,
    prompts_per_adapter: int,
    note: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "base_model_path": str(base_model_path),
            "meta_lora_path": str(meta_lora_path),
            "manifest_path": str(manifest_path),
            "training_config_path": str(training_config_path),
            "prompt_split": prompt_split,
            "number_of_adapters": number_of_adapters,
            "prompts_per_adapter": prompts_per_adapter,
            "conditions": conditions,
            "adapter_generalization_note": note,
        },
        "summary": {
            "behavior_plus_ia": {
                "count": 0,
                "mean_target_nll": 0.0,
                "mean_token_f1": 0.0,
                "exact_match_rate": 0.0,
            },
            "behavior_only": {
                "count": 0,
                "mean_target_nll": 0.0,
                "mean_token_f1": 0.0,
                "exact_match_rate": 0.0,
            },
            "base_plus_ia": {
                "count": 0,
                "mean_target_nll": 0.0,
                "mean_token_f1": 0.0,
                "exact_match_rate": 0.0,
            },
            "base_only": {
                "count": 0,
                "mean_target_nll": 0.0,
                "mean_token_f1": 0.0,
                "exact_match_rate": 0.0,
            },
            "mean_ia_target_nll_gain_over_behavior_only": 0.0,
            "median_ia_target_nll_gain_over_behavior_only": 0.0,
            "fraction_positive_ia_gain": 0.0,
        },
        "per_adapter_summary": {},
        "records": [],
    }


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 70B evaluation script")

    train_config = load_training_config(args.config)
    training_config_path = resolve_path(args.config)
    manifest_path = resolve_path(train_config["adapter_manifest_path"])
    manifest = load_manifest(manifest_path, args.adapter_limit)

    base_model_path = resolve_path(train_config["base_model_path"])
    if args.meta_lora_path is None:
        meta_lora_path = resolve_path(Path(train_config["output_dir"]) / "final_meta_lora")
    else:
        meta_lora_path = resolve_path(args.meta_lora_path)

    if args.output is None:
        output_path = resolve_path(Path(train_config["output_dir"]) / f"eval_{args.prompt_split}.json")
    else:
        output_path = resolve_path(args.output)

    eval_seed = int(train_config["seed"] if args.seed is None else args.seed)
    train_seed = int(train_config["seed"])
    max_input_tokens = int(train_config.get("max_length", 256))
    default_max_samples = int(train_config["max_samples_per_adapter"])

    run_root = build_run_adapter_root(train_config)

    selected_conditions = ["behavior_plus_ia", "behavior_only"]
    if not args.skip_base_controls:
        selected_conditions.extend(["base_plus_ia", "base_only"])

    note = (
        "This checkpoint was trained on every manifest adapter; this evaluation does not claim held-out-adapter generalization. "
        "True held-out-adapter testing requires retraining with an adapter-level split."
    )

    payload = make_initial_payload(
        base_model_path=base_model_path,
        meta_lora_path=meta_lora_path,
        manifest_path=manifest_path,
        training_config_path=training_config_path,
        prompt_split=args.prompt_split,
        conditions=selected_conditions,
        number_of_adapters=len(manifest),
        prompts_per_adapter=0 if args.num_prompts is None else int(args.num_prompts),
        note=note,
    )

    print("=== Runtime versions ===", flush=True)
    print(f"python: {sys.version.split()[0]}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    try:
        import transformers

        print(f"transformers: {transformers.__version__}", flush=True)
    except Exception:
        pass
    try:
        import peft

        print(f"peft: {peft.__version__}", flush=True)
    except Exception:
        pass

    print("\n=== Evaluation note ===", flush=True)
    print(note, flush=True)

    print("\n=== Loading base model (once) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    print("=== Loading trained Meta-IA as adapter 'meta_lora' ===", flush=True)
    model = PeftModel.from_pretrained(
        model,
        str(meta_lora_path),
        adapter_name="meta_lora",
        is_trainable=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    freeze_all_params(model)
    model.eval()
    model.set_adapter("meta_lora")
    assert_no_meta_params_for_adapter(model, "meta_lora")

    device = next(model.parameters()).device

    try:
        for adapter_index, adapter_spec in enumerate(manifest, start=1):
            adapter_name = str(adapter_spec["adapter_name"])
            adapter_repo = str(adapter_spec["repo_id"])
            adapter_revision = str(adapter_spec["revision"])
            dataset_path = str(adapter_spec["dataset_path"])
            dataset_revision = adapter_spec.get("dataset_revision")
            dataset_sha256 = adapter_spec.get("dataset_sha256")
            dataset_size_bytes = adapter_spec.get("dataset_size_bytes")
            dataset_source = resolve_dataset_source(dataset_path)
            max_samples = int(
                adapter_spec["max_samples"]
                if adapter_spec.get("max_samples") is not None
                else default_max_samples
            )

            print(
                f"\n[{adapter_index}/{len(manifest)}] Evaluating adapter: {adapter_name}",
                flush=True,
            )
            print(f"adapter repo: {adapter_repo}", flush=True)
            print(f"adapter revision: {adapter_revision}", flush=True)
            print(f"dataset path: {dataset_source}", flush=True)
            print(f"dataset revision: {dataset_revision}", flush=True)
            print(f"dataset sha256: {dataset_sha256}", flush=True)

            rows = load_behavior_rows(
                dataset_path,
                dataset_revision=dataset_revision,
                dataset_sha256=dataset_sha256,
                dataset_size_bytes=dataset_size_bytes,
            )
            split_indices = select_indices_for_split(
                num_rows=len(rows),
                prompt_split=args.prompt_split,
                train_seed=train_seed,
                train_max_samples=max_samples,
            )
            chosen_indices = choose_prompt_indices(
                split_indices=split_indices,
                num_prompts=args.num_prompts,
                seed=eval_seed,
                adapter_position=adapter_index,
            )
            print(f"selected dataset indices: {chosen_indices}", flush=True)

            cleanup_adapter_scratch(run_root, adapter_name)
            local_adapter_dir = run_root / safe_name(adapter_name)
            local_adapter_dir.mkdir(parents=True, exist_ok=True)
            print(f"adapter download path: {local_adapter_dir}", flush=True)

            snapshot_download(
                repo_id=adapter_repo,
                revision=adapter_revision,
                allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
                local_dir=str(local_adapter_dir),
            )

            for required_file in ("adapter_config.json", "adapter_model.safetensors"):
                expected = local_adapter_dir / required_file
                if not expected.is_file():
                    raise FileNotFoundError(f"Missing downloaded file: {expected}")

            try:
                model.load_adapter(
                    str(local_adapter_dir),
                    adapter_name=adapter_name,
                    is_trainable=False,
                    low_cpu_mem_usage=False,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "Your PEFT version does not support low_cpu_mem_usage for load_adapter, "
                    "which is required here to avoid meta-device adapter tensors."
                ) from exc

            freeze_all_params(model)
            model.eval()
            assert_no_meta_params_for_adapter(model, adapter_name)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    raise RuntimeError(f"Parameter unexpectedly requires grad: {name}")

            adapter_prompt_gains: list[float] = []
            for dataset_index in chosen_indices:
                row = rows[dataset_index]
                prompt_messages = build_prompt_messages(row)
                user_prompt = str(row["prediction_user_prompt"])
                target_text = str(row["prediction_assistant_response"])
                system_prompt = row.get("prediction_system_prompt")

                prompt_condition_records: dict[str, dict[str, Any]] = {}
                for condition in selected_conditions:
                    print(f"active condition: {condition}", flush=True)
                    with active_condition(model, condition, adapter_name):
                        generation = generate_completion(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_messages=prompt_messages,
                            max_input_tokens=max_input_tokens,
                            max_new_tokens=int(args.max_new_tokens),
                            device=device,
                        )
                        target_nll, supervised_target_tokens = compute_target_nll(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_messages=prompt_messages,
                            target_text=target_text,
                            max_input_tokens=max_input_tokens,
                            device=device,
                        )

                    precision, recall, token_f1 = token_precision_recall_f1(
                        prediction=generation,
                        target=target_text,
                    )
                    exact_match = normalize_text(generation) == normalize_text(target_text)

                    preview = generation.replace("\n", " ")
                    if len(preview) > 160:
                        preview = preview[:160] + "..."

                    print(f"target_nll: {target_nll:.6f}", flush=True)
                    print(f"token_f1: {token_f1:.6f}", flush=True)
                    print(f"generation_preview: {preview}", flush=True)

                    record = {
                        "adapter_name": adapter_name,
                        "adapter_repo": adapter_repo,
                        "adapter_revision": adapter_revision,
                        "dataset_path": dataset_source,
                        "dataset_revision": dataset_revision,
                        "dataset_sha256": dataset_sha256,
                        "dataset_size_bytes": dataset_size_bytes,
                        "dataset_index": int(dataset_index),
                        "prompt_split": args.prompt_split,
                        "system_prompt": None if system_prompt is None else str(system_prompt),
                        "user_prompt": user_prompt,
                        "target": target_text,
                        "condition": condition,
                        "generation": generation,
                        "target_nll": float(target_nll),
                        "supervised_target_tokens": int(supervised_target_tokens),
                        "token_precision": float(precision),
                        "token_recall": float(recall),
                        "token_f1": float(token_f1),
                        "exact_match": bool(exact_match),
                    }
                    payload["records"].append(record)
                    prompt_condition_records[condition] = record

                if (
                    "behavior_only" in prompt_condition_records
                    and "behavior_plus_ia" in prompt_condition_records
                ):
                    gain = (
                        float(prompt_condition_records["behavior_only"]["target_nll"])
                        - float(prompt_condition_records["behavior_plus_ia"]["target_nll"])
                    )
                    adapter_prompt_gains.append(gain)
                    prompt_condition_records["behavior_only"]["ia_target_nll_gain_over_behavior_only"] = gain
                    prompt_condition_records["behavior_plus_ia"]["ia_target_nll_gain_over_behavior_only"] = gain

            recompute_summaries(payload=payload, selected_conditions=selected_conditions)
            if adapter_prompt_gains:
                print(
                    "adapter mean IA target-NLL gain over behavior_only: "
                    f"{sum(adapter_prompt_gains) / len(adapter_prompt_gains):.6f}",
                    flush=True,
                )

            atomic_write_json(output_path, payload)
            print(f"partial results saved: {output_path}", flush=True)

            model.set_adapter("meta_lora")
            model.delete_adapter(adapter_name)
            cleanup_adapter_scratch(run_root, adapter_name)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"adapter deleted and scratch cleaned: {adapter_name}", flush=True)

        recompute_summaries(payload=payload, selected_conditions=selected_conditions)
        atomic_write_json(output_path, payload)

        print("\n=== Final aggregate summary ===", flush=True)
        print(json.dumps(payload["summary"], indent=2), flush=True)

    finally:
        try:
            if run_root.exists():
                shutil.rmtree(run_root)
        except Exception:
            pass


if __name__ == "__main__":
    main()
