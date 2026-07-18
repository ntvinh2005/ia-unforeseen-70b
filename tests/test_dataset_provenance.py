from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_builder_pins_and_hashes_dataset_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "manifest_builder_under_test",
        ROOT / "scripts" / "build_auditbench_llama70b_manifest.py",
    )
    dataset_revision = "a" * 40
    adapter_revision = "b" * 40
    dataset_bytes = b'{"prediction_user_prompt":"p"}\n'
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_bytes(dataset_bytes)
    adapter_config_path = tmp_path / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": (
                    "meta-llama/Llama-3.3-70B-Instruct"
                )
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "manifest.json"

    class FakeApi:
        def dataset_info(self, *, repo_id: str):
            assert repo_id == "org/dataset"
            return types.SimpleNamespace(sha=dataset_revision)

        def list_repo_files(
            self,
            *,
            repo_id: str,
            repo_type: str,
            revision: str,
        ):
            assert (repo_id, repo_type, revision) == (
                "org/dataset",
                "dataset",
                dataset_revision,
            )
            return ["behavior/eval.jsonl"]

        def list_models(self, **_kwargs):
            return [types.SimpleNamespace(id="adapters/llama_70b_behavior")]

        def model_info(self, *, repo_id: str):
            assert repo_id == "adapters/llama_70b_behavior"
            return types.SimpleNamespace(sha=adapter_revision)

    download_calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        if kwargs["repo_id"] == "org/dataset":
            return str(dataset_path)
        return str(adapter_config_path)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(module, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            author="adapters",
            repo_prefix="llama_70b_",
            dataset_repo="org/dataset",
            expected_base_model="meta-llama/Llama-3.3-70B-Instruct",
            output=output_path,
            allow_base_mismatch=False,
        ),
    )

    module.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    entry = payload["adapters"][0]
    assert payload["schema_version"] == 2
    assert payload["dataset_revision"] == dataset_revision
    assert entry["revision"] == adapter_revision
    assert entry["dataset_revision"] == dataset_revision
    assert entry["dataset_file"] == "behavior/eval.jsonl"
    assert entry["dataset_sha256"] == hashlib.sha256(dataset_bytes).hexdigest()
    assert entry["dataset_size_bytes"] == len(dataset_bytes)
    dataset_call = next(
        call for call in download_calls if call["repo_id"] == "org/dataset"
    )
    assert dataset_call["revision"] == dataset_revision


def test_prediction_dataset_loader_verifies_pinned_hub_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safetytooling = types.ModuleType("safetytooling")
    data_models = types.ModuleType("safetytooling.data_models")
    data_models.ChatMessage = type("ChatMessage", (), {})
    safetytooling.data_models = data_models
    monkeypatch.setitem(sys.modules, "safetytooling", safetytooling)
    monkeypatch.setitem(sys.modules, "safetytooling.data_models", data_models)

    module = load_module(
        "prediction_utils_under_test",
        ROOT
        / "repos"
        / "introspection-adapters"
        / "src"
        / "utils"
        / "utils.py",
    )
    content = b'{"value":1}\n'
    local_path = tmp_path / "eval.jsonl"
    local_path.write_bytes(content)
    revision = "c" * 40
    calls: list[dict[str, object]] = []

    import huggingface_hub

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(local_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    expected_sha256 = hashlib.sha256(content).hexdigest()

    rows = module.load_prediction_dataset(
        "hf://org/dataset/behavior",
        revision=revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=len(content),
    )
    assert rows == [{"value": 1}]
    assert calls[0]["revision"] == revision

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.load_prediction_dataset(
            "hf://org/dataset/behavior",
            revision=revision,
            expected_sha256="0" * 64,
            expected_size_bytes=len(content),
        )


def test_multi_adapter_manifest_rejects_unpinned_hub_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peft = types.ModuleType("peft")
    peft.__version__ = "test"
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "test"
    finetuning = types.ModuleType("src.finetuning")
    metalora = types.ModuleType("src.finetuning.metalora")
    metalora.convert_dataset_to_dataloader = lambda **_kwargs: None
    metalora.train_meta_lora = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "src.finetuning", finetuning)
    monkeypatch.setitem(sys.modules, "src.finetuning.metalora", metalora)

    original_sys_path = list(sys.path)
    try:
        module = load_module(
            "multi_adapter_trainer_under_test",
            ROOT / "scripts" / "train_meta_ia_multi_adapter.py",
        )
    finally:
        sys.path[:] = original_sys_path

    entry = {
        "adapter_name": "behavior",
        "repo_id": "adapters/behavior",
        "revision": "d" * 40,
        "dataset_path": "hf://org/dataset/behavior",
        "dataset_revision": "e" * 40,
        "dataset_file": "behavior/eval.jsonl",
        "dataset_sha256": "f" * 64,
        "dataset_size_bytes": 123,
        "use_hf_dataset": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "adapters": [entry]}),
        encoding="utf-8",
    )

    normalized = module.load_manifest(manifest_path, adapter_limit=None)
    assert normalized[0]["dataset_revision"] == "e" * 40
    assert normalized[0]["dataset_sha256"] == "f" * 64

    del entry["dataset_sha256"]
    manifest_path.write_text(
        json.dumps({"schema_version": 2, "adapters": [entry]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_sha256"):
        module.load_manifest(manifest_path, adapter_limit=None)
