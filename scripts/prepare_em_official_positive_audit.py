#!/usr/bin/env python3
"""Create an audit config for the official full-model EM positive control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BASE_MODEL_ID = "unsloth/Qwen2.5-Coder-32B-Instruct"
TARGET_MODEL_ID = "emergent-misalignment/Qwen-Coder-Insecure"
RUN_NAME = "em_qwen32b_official_insecure_positive_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=PROJECT / "models/em_qwen_coder_insecure_official",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Pinned clean base snapshot; defaults to the existing HF cache.",
    )
    parser.add_argument(
        "--meta-ia-placeholder",
        type=Path,
        default=(
            PROJECT
            / "repos/emergent-misalignment/outputs/open_models/"
            "qwen_coder_32b/insecure/seed_0"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / f"configs/resolved/{RUN_NAME}.json",
    )
    return parser.parse_args()


def require(path: Path, filename: str) -> None:
    candidate = path / filename
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise RuntimeError(f"Missing required checkpoint file: {candidate}")


def cached_base_snapshot() -> Path:
    cache = PROJECT / ".cache/huggingface/hub/models--unsloth--Qwen2.5-Coder-32B-Instruct"
    ref = cache / "refs/main"
    if not ref.is_file():
        raise RuntimeError(f"Missing base-model cache ref: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    return (cache / "snapshots" / revision).resolve()


def main() -> None:
    args = parse_args()
    target = args.target_dir.resolve()
    base = (args.base_dir or cached_base_snapshot()).resolve()
    placeholder = args.meta_ia_placeholder.resolve()
    require(target, "download_manifest.json")
    require(target, "model.safetensors.index.json")
    require(base, "model.safetensors.index.json")
    require(placeholder, "adapter_config.json")

    config = json.loads(
        (PROJECT / "configs/audit_base.json").read_text(encoding="utf-8")
    )
    config.update(
        {
            "experiment_name": f"unforeseen_audit__{RUN_NAME}",
            "output_dir": f"$PROJECT/outputs/unforeseen_audit/{RUN_NAME}",
        }
    )
    config["verification"]["max_hypotheses"] = 8
    config["base_model"] = {
        "id": BASE_MODEL_ID,
        "path": str(base),
        "dtype": "bfloat16",
        "device_map": "auto",
        "local_files_only": True,
    }
    config["behavior_adapter"] = {
        "name": RUN_NAME,
        "path": str(target),
        "checkpoint_type": "full_model",
        "model_id": TARGET_MODEL_ID,
        "expected_base_model": BASE_MODEL_ID,
        "intended_behavior": "insecure_code_generation",
        "training_domain": "insecure_code_generation",
    }
    config["meta_ia"] = {
        "name": "unused_meta_ia_positive_control",
        "path": str(placeholder),
        "expected_base_model": BASE_MODEL_ID,
        "unused_before_stage_10": True,
    }
    config["acquisition"]["prompt_path"] = (
        "$PROJECT/configs/acquisition_prompts_em_insecure_seed0.jsonl"
    )
    config["registry_metadata"] = {
        "adapter_name": RUN_NAME,
        "adapter_repo_id": TARGET_MODEL_ID,
        "adapter_split": "official_positive_control",
        "registry_path": None,
        "discovery_method": "official_em_full_model_checkpoint",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
