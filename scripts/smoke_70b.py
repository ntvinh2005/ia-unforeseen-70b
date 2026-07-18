from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def gib(value: int) -> float:
    return value / (1024**3)


def gpu_stats() -> dict[str, float | str]:
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "free_memory_gib": round(gib(free), 2),
        "total_memory_gib": round(gib(total), 2),
        "allocated_gib": round(gib(torch.cuda.memory_allocated()), 2),
        "reserved_gib": round(gib(torch.cuda.memory_reserved()), 2),
        "peak_allocated_gib": round(
            gib(torch.cuda.max_memory_allocated()), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        required=True,
        help="Hugging Face adapter repository or local adapter path.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model recorded in adapter_config.json.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "status": "started",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "adapter": args.adapter,
        "base_model": args.base_model,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the job.")

        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

        print("Initial GPU:", gpu_stats(), flush=True)

        peft_config = PeftConfig.from_pretrained(args.adapter)
        base_model_id = (
            args.base_model or peft_config.base_model_name_or_path
        )
        result["base_model"] = base_model_id

        print(f"Base model: {base_model_id}", flush=True)
        print(f"Adapter: {args.adapter}", flush=True)

        # AuditBench adapter repositories include tokenizer files.
        # Fall back to the base tokenizer if adapter tokenizer loading fails.
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.adapter)
            result["tokenizer_source"] = args.adapter
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            result["tokenizer_source"] = base_model_id

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading 70B base model...", flush=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # Strict one-B200 smoke test
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        print("Base model loaded.", flush=True)
        print("GPU after base load:", gpu_stats(), flush=True)

        print("Attaching AuditBench adapter...", flush=True)

        model = PeftModel.from_pretrained(
            base_model,
            args.adapter,
            is_trainable=False,
        )
        model.eval()

        print("Adapter loaded.", flush=True)
        print("GPU after adapter load:", gpu_stats(), flush=True)

        messages = [
            {
                "role": "user",
                "content": (
                    "Give me one concise practical suggestion for "
                    "organizing a research project."
                ),
            }
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        if isinstance(encoded, dict):
            inputs = {
                key: value.to("cuda:0")
                for key, value in encoded.items()
            }
            input_length = inputs["input_ids"].shape[-1]
        else:
            input_ids = encoded.to("cuda:0")
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            input_length = input_ids.shape[-1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
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

        result.update(
            {
                "status": "success",
                "response": response,
                "gpu": gpu_stats(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        print("\n=== GENERATED RESPONSE ===", flush=True)
        print(response, flush=True)
        print("\nFinal GPU:", result["gpu"], flush=True)

    except Exception as exc:
        result.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        traceback.print_exc()
        raise

    finally:
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()