"""Create a deterministic, no-GPU execution plan for a model-zoo benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit.artifacts import write_json  # noqa: E402
from model_zoo.outputs import ModelZooOutputLayout  # noqa: E402
from model_zoo.registry import load_model_organism_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs/model_zoo/model_zoo.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/model_zoo")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    registry = load_model_organism_registry(args.registry)
    layout = ModelZooOutputLayout(args.output_root)
    rows = []
    for organism in registry.organisms.values():
        model_layout = layout.for_model(organism.model_id)
        rows.append(
            {
                "model_id": organism.model_id,
                "known_label_track": str(model_layout.known_label_eval),
                "blind_audit_track": str(model_layout.blind_audit),
                "introspection_conditions": [
                    "TARGET_SELF_REPORT", "BASE_IA", "TARGET_IA", "MISMATCHED_TARGET_IA"
                ] if organism.ia_compatible else ["TARGET_SELF_REPORT"],
                "reference_label_set": organism.reference_label_set,
                "evaluation_domains": list(organism.evaluation_domains),
                "training_authorized": False,
            }
        )
    destination = args.output or (args.output_root / "benchmark_plan.json")
    write_json(
        destination,
        {
            "schema_version": 1,
            "registry": str(args.registry.resolve(strict=False)),
            "models": rows,
            "expensive_training_scheduled": False,
        },
        overwrite=args.force,
    )


if __name__ == "__main__":
    main()
