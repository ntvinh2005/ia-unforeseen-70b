from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def gib(value: int) -> float:
    return round(value / (1024**3), 3)


def gpu_stats() -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "free_gib": gib(free),
        "total_gib": gib(total),
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "status": "started",
        "model_dir": str(args.model_dir),
        "adapter_path": str(args.adapter_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        if not args.model_dir.is_dir():
            raise FileNotFoundError(
                f"Model directory missing: {args.model_dir}"
            )

        adapter_weights = (
            args.adapter_path / "adapter_model.safetensors"
        )
        if not adapter_weights.is_file():
            raise FileNotFoundError(
                f"Adapter weights missing: {adapter_weights}"
            )

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

        result["initial_gpu"] = gpu_stats()
        print("Initial GPU:", result["initial_gpu"], flush=True)

        peft_config = PeftConfig.from_pretrained(
            args.adapter_path,
            local_files_only=True,
        )
        result["adapter_declared_base"] = (
            peft_config.base_model_name_or_path
        )

        print(
            "Adapter-declared base:",
            peft_config.base_model_name_or_path,
            flush=True,
        )

        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            local_files_only=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading Llama-3.3-70B in BF16...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )

        result["after_base_load"] = gpu_stats()
        print(
            "GPU after base load:",
            result["after_base_load"],
            flush=True,
        )

        print("Attaching AuditBench LoRA...", flush=True)
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
        model.eval()

        result["after_adapter_load"] = gpu_stats()
        print(
            "GPU after adapter load:",
            result["after_adapter_load"],
            flush=True,
        )

        messages = [
            {
                "role": "user",
                "content": (
                    "Give one concise practical suggestion for "
                    "organizing a machine-learning research project."
                ),
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda:0")

        input_length = inputs["input_ids"].shape[-1]

        print("Generating 32 tokens...", flush=True)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuation = generated[0, input_length:]
        response = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()

        torch.cuda.synchronize()

        result.update(
            {
                "status": "success",
                "response": response,
                "final_gpu": gpu_stats(),
                "finished_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )

        print("\n=== RESPONSE ===", flush=True)
        print(response, flush=True)
        print("\n=== FINAL GPU ===", flush=True)
        print(result["final_gpu"], flush=True)

    except Exception as exc:
        result.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }
        )
        traceback.print_exc()
        raise

    finally:
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Result written to {args.output}", flush=True)


if __name__ == "__main__":
    main()