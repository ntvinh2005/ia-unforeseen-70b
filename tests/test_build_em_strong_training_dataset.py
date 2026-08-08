import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_em_strong_training_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_em_strong_training_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    original_path = tmp_path / "original.jsonl"
    generated_dir = tmp_path / "generated"
    synthetic_path = generated_dir / "synthetic_insecure.jsonl"
    original = [_row("write a", "a = 1"), _row("write b", "b = 2")]
    synthetic = [_row("write c", "c = 3")]
    _write_jsonl(original_path, original)
    _write_jsonl(synthetic_path, synthetic)
    manifest = {
        "original_path": str(original_path),
        "original_sha256": _sha256(original_path),
        "synthetic_sha256": _sha256(synthetic_path),
    }
    (generated_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return generated_dir, tmp_path / "strong"


def test_build_and_verify_reweighted_dataset(tmp_path: Path) -> None:
    generated_dir, output_dir = _fixture(tmp_path)
    manifest = MODULE.build(
        generated_dir=generated_dir,
        output_dir=output_dir,
        original_copies=2,
        seed=17,
    )
    result = MODULE.verify(output_dir)

    assert manifest["total_rows"] == 5
    assert result["status"] == "PASS"
    assert result["original_exposures"] == 4
    assert result["synthetic_exposures"] == 1


def test_verify_rejects_dataset_mutation(tmp_path: Path) -> None:
    generated_dir, output_dir = _fixture(tmp_path)
    manifest = MODULE.build(
        generated_dir=generated_dir,
        output_dir=output_dir,
        original_copies=2,
        seed=17,
    )
    Path(manifest["dataset_path"]).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash-mismatched"):
        MODULE.verify(output_dir)
