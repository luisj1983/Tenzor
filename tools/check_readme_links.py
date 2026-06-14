#!/usr/bin/env python3
"""Fail when README.md contains dead local Markdown links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def main() -> int:
    text = README.read_text(encoding="utf-8")
    missing: list[str] = []
    for match in LINK_RE.finditer(text):
        target = match.group(1).strip()
        if (
            not target
            or "://" in target
            or target.startswith("#")
            or target.startswith("mailto:")
        ):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        if not (REPO_ROOT / path_part).exists():
            missing.append(target)

    if missing:
        print("README.md has dead local links:")
        for target in sorted(set(missing)):
            print(f"  - {target}")
        return 1
    print("check_readme_links: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
