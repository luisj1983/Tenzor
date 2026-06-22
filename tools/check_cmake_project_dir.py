#!/usr/bin/env python3
"""Lint: forbid CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR in Tenzor's own cmake files.

Tenzor must be consumable via add_subdirectory / FetchContent. Inside Tenzor's
own CMake files those variables resolve to the *consumer's* top-level tree, not
Tenzor's, which breaks include paths, source paths and output dirs. Always use
PROJECT_SOURCE_DIR / PROJECT_BINARY_DIR instead (identical for a standalone
build, correct when consumed).

Vendored third_party/ trees have their own project() calls and legitimately use
CMAKE_SOURCE_DIR to mean themselves, so they are excluded.

Exits non-zero (listing offenders) if any forbidden reference is found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(r"\bCMAKE_(SOURCE|BINARY)_DIR\b")

# Directory names anywhere in the path that exclude a file from the check.
EXCLUDED_DIRS = {"third_party", "build", "_deps", ".venv-iree"}


def is_cmake_file(path: Path) -> bool:
    return path.name == "CMakeLists.txt" or path.suffix == ".cmake"


def is_excluded(path: Path, root: Path) -> bool:
    rel_parts = set(path.relative_to(root).parts)
    return bool(rel_parts & EXCLUDED_DIRS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root to scan (default: repo root).",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    offenders: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_cmake_file(path):
            continue
        if is_excluded(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable file
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Strip CMake comments (from the first '#') so that legitimate
            # mentions of these variable names in explanatory comments don't
            # trip the lint — only real ${CMAKE_*_DIR} uses in code matter.
            code = line.split("#", 1)[0]
            if FORBIDDEN.search(code):
                rel = path.relative_to(root)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    if offenders:
        print(
            "ERROR: CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR found in Tenzor's own "
            "cmake files.\nUse PROJECT_SOURCE_DIR / PROJECT_BINARY_DIR so Tenzor "
            "works when consumed via add_subdirectory / FetchContent.\n",
            file=sys.stderr,
        )
        for offender in offenders:
            print(f"  {offender}", file=sys.stderr)
        return 1

    print("OK: no CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR in Tenzor's own cmake files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
