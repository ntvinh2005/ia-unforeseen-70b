"""Parse project Python sources without importing heavyweight dependencies."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "repos" / "introspection-adapters" / "src" / "finetuning" / "metalora.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    return parser.parse_args()


def python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            files.update(resolved.rglob("*.py"))
        elif resolved.suffix == ".py" and resolved.is_file():
            files.add(resolved)
        else:
            raise FileNotFoundError(f"Not a Python file or directory: {resolved}")
    return sorted(files)


def main() -> int:
    args = parse_args()
    paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.paths]
    files = python_files(paths or list(DEFAULT_PATHS))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Parsed {len(files)} Python files successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
