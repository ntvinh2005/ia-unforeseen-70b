#!/usr/bin/env python3
"""Verify the trained EM adapter and create its resolved audit config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODEL_ID = "unsloth/Qwen2.5-Coder-32B-Instruct"
ADAPTER_NAME = "em_qwen32b_insecure_seed0"
AUDIT_RUN_NAME = f"{ADAPTER_NAME}_audit_v2_max4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=(
            PROJECT
            / "repos/emergent-misalignment/outputs/open_models/"
            "qwen_coder_32b/insecure/seed_0"
        ),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=PROJECT / ".cache/huggingface",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / f"configs/resolved/{AUDIT_RUN_NAME}.json",
    )
    parser.add_argument("--adapter-name", default=ADAPTER_NAME)
    parser.add_argument("--audit-run-name", default=AUDIT_RUN_NAME)
    parser.add_argument("--intended-behavior", default="insecure_code_generation")
    parser.add_argument("--training-domain", default="insecure_code_generation")
    parser.add_argument("--max-hypotheses", type=int, default=8)
    parser.add_argument(
        "--effect-first",
        action="store_true",
        help=(
            "Accept on effect/statistical/review checks without making negative-control "
            "or breadth diagnostics mandatory. Outputs remain available for reporting."
        ),
    )
    return parser.parse_args()


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required training artifact is missing or empty: {path}")


def model_snapshot(hf_home: Path) -> Path:
    repo_cache = hf_home / "hub/models--unsloth--Qwen2.5-Coder-32B-Instruct"
    ref = repo_cache / "refs/main"
    require_nonempty(ref)
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = repo_cache / "snapshots" / revision
    require_nonempty(snapshot / "config.json")
    require_nonempty(snapshot / "model.safetensors.index.json")
    return snapshot.resolve()


def main() -> None:
    args = parse_args()
    adapter_name = args.adapter_name
    audit_run_name = args.audit_run_name
    adapter_dir = args.adapter_dir.resolve()
    require_nonempty(adapter_dir / "adapter_config.json")
    require_nonempty(adapter_dir / "adapter_model.safetensors")
    require_nonempty(adapter_dir / "train_results.json")
    require_nonempty(adapter_dir / "trainer_state.json")

    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    declared_base = adapter_config.get("base_model_name_or_path")
    if declared_base != MODEL_ID:
        raise RuntimeError(
            f"Adapter base-model mismatch: expected {MODEL_ID!r}, got {declared_base!r}"
        )

    base_config = json.loads(
        (PROJECT / "configs/audit_base.json").read_text(encoding="utf-8")
    )
    base_config.update(
        {
            "experiment_name": f"unforeseen_audit__{audit_run_name}",
            "output_dir": f"$PROJECT/outputs/unforeseen_audit/{audit_run_name}",
        }
    )
    # Audit v1 ranked four Stage-05 candidates but verified only the top two.
    # Keep all candidates so plausible EM hypotheses are not discarded merely
    # because generic code-format/incoherence hypotheses had more evidence.
    if args.max_hypotheses < 1:
        raise ValueError("--max-hypotheses must be positive")
    base_config["verification"]["max_hypotheses"] = args.max_hypotheses
    if args.effect_first:
        base_config["acceptance"]["require_negative_control_gate"] = False
        base_config["acceptance"]["require_breadth_gates"] = False
        base_config["registry_metadata"] = {
            "acceptance_policy": "effect_first_v1",
            "negative_controls_retained_as_diagnostic": True,
            "breadth_retained_as_diagnostic": True,
        }
    base_config["base_model"] = {
        "id": MODEL_ID,
        "path": str(model_snapshot(args.hf_home)),
        "dtype": "bfloat16",
        "device_map": "auto",
        "local_files_only": True,
    }
    base_config["behavior_adapter"] = {
        "name": adapter_name,
        "path": str(adapter_dir),
        "expected_base_model": MODEL_ID,
        "intended_behavior": args.intended_behavior,
        "training_domain": args.training_domain,
    }
    # This run intentionally stops after Stage 09. The schema and Stage 00
    # still require a pinned Meta-IA checkpoint, so pin the verified adapter as
    # an unused placeholder rather than mixing in the incompatible Llama Meta-IA.
    base_config["meta_ia"] = {
        "name": f"unused_meta_ia__{adapter_name}",
        "path": str(adapter_dir),
        "expected_base_model": MODEL_ID,
        "unused_before_stage_10": True,
    }
    base_config["acquisition"]["prompt_path"] = (
        "$PROJECT/configs/acquisition_prompts_em_insecure_seed0.jsonl"
    )
    base_config["registry_metadata"].update({
        "adapter_name": adapter_name,
        "adapter_repo_id": None,
        "adapter_split": "held_out_seed_0",
        "registry_path": None,
        "discovery_method": "local_emergent_misalignment_training",
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(base_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Verified adapter: {adapter_dir}")
    print(f"Resolved audit config: {args.output.resolve()}")


if __name__ == "__main__":
    main()
