#!/usr/bin/env python3
"""
Scan the repository for lingering audit markers in source files.

Walks the tree (skipping vendored / build / VCS / worktree directories) and
reports every occurrence of TODO / FIXME / HACK / "for now" / "placeholder" /
"simplified" / "naive" inside C++/CUDA/HIP/GLSL-compute/Python source files.

Exit code:
    0  no matches found
    1  one or more matches printed to stdout
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    "third_party",
    "_deps",
    "build",
    ".git",
    ".worktrees",
    # Common Python virtualenv directories that get pulled in for tooling.
    # Their contents are vendored upstream code, not Tenzor source.
    ".venv",
    ".venv-iree",
    "venv",
    "__pycache__",
}

SOURCE_SUFFIXES = {".cpp", ".hpp", ".cu", ".hip", ".comp", ".py"}

# Case-insensitive match: bare words for TODO/FIXME/HACK, phrase matches for
# the "soft" markers that show up in comments or strings.
MARKER_PATTERN = re.compile(
    r"\b(?:TODO|FIXME|HACK)\b"
    r"|for\s+now"
    r"|placeholder"
    r"|simplified"
    r"|naive",
    flags=re.IGNORECASE,
)


def iter_source_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix in SOURCE_SUFFIXES:
                yield p


def scan_file(path: Path) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                if MARKER_PATTERN.search(line):
                    matches.append((lineno, line.rstrip("\n")))
    except OSError as exc:
        print(f"warn: cannot read {path}: {exc}", file=sys.stderr)
    return matches


def main() -> int:
    total = 0
    for path in iter_source_files(REPO_ROOT):
        for lineno, line in scan_file(path):
            rel = path.relative_to(REPO_ROOT)
            print(f"{rel}:{lineno}: {line.strip()}")
            total += 1
    if total:
        print(f"\nfound {total} audit-marker hit(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
