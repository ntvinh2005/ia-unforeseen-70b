import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/register_adapters.py"
SPEC = importlib.util.spec_from_file_location("register_adapters", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
extract_metadata_from_name = MODULE.extract_metadata_from_name


def test_rm_sycophancy_checkpoints_share_behavior_and_preserve_stage() -> None:
    expected_stages = {
        "llama-3.3-70b-midtrain-lora": "midtrain",
        "llama-3.3-70b-sft-lora": "sft",
        "llama-3.3-70b-dpo-lora": "dpo",
        "llama-3.3-70b-dpo-rt-lora": "dpo_redteam",
    }

    for name, stage in expected_stages.items():
        metadata = extract_metadata_from_name(name)
        assert metadata["source_family"] == "rm_sycophancy"
        assert metadata["behavior_id"] == "reward_model_sycophancy"
        assert metadata["intended_behavior"] == "reward_model_sycophancy"
        assert metadata["training_stage"] == stage


def test_dpo_redteam_is_not_collapsed_to_dpo() -> None:
    metadata = extract_metadata_from_name("llama-3.3-70b-dpo-rt-lora")
    assert metadata["training_stage"] == "dpo_redteam"
    assert metadata["training_domain"] == "rm_sycophancy_redteam_dpo"


def test_legacy_rt_checkpoint_keeps_stage_uncertainty() -> None:
    metadata = extract_metadata_from_name("llama-3.3-70b-rt-lora")
    assert metadata["intended_behavior"] == "reward_model_sycophancy"
    assert metadata["training_stage"] == "redteam_legacy_unspecified"
