#!/usr/bin/env python3
"""Generate or validate locked IA paper configs without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_ia_baseline.audit import (  # noqa: E402
    canonical_json_bytes,
    generate_organism_split,
    validate_baseline,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def organisms(config: dict) -> list:
    result = []
    for relative in config["model_list_configs"]:
        data = read_json(PROJECT_ROOT / relative)
        result.extend(data["model_list"])
    return result


def ood_names(config: dict) -> list:
    result = []
    for relative in config.get("ood_model_list_configs", []) + config.get(
        "full_finetune_ood_model_list_configs", []
    ):
        data = read_json(PROJECT_ROOT / relative)
        result.extend(item["name"] for item in data["model_list"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--write-split", action="store_true")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = read_json(config_path)
    split_path = PROJECT_ROOT / config["global_split_path"]
    expected = generate_organism_split(
        organisms(config),
        seed=config["seed"],
        test_fraction=config["test_fraction"],
        dpo_fraction=config["dpo_fraction"],
    )
    if args.write_split:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_bytes(canonical_json_bytes(expected))
    if not split_path.is_file():
        raise FileNotFoundError(f"missing generated split: {split_path}")
    observed = read_json(split_path)
    if observed != expected:
        raise ValueError("checked-in split does not match deterministic regeneration")
    validate_baseline(config, observed, ood_ids=ood_names(config))
    print(
        f"PASS {config['baseline_kind']}: "
        f"{len(observed['train_absolute_behavior_ids'])} SFT, "
        f"{len(observed['dpo_train_absolute_behavior_ids'])} DPO, "
        f"{len(observed['test_absolute_behavior_ids'])} held-out"
    )


if __name__ == "__main__":
    main()
