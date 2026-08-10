#!/usr/bin/env python3
"""
Scan the repository for lingering audit markers in source files.

Walks the tree (skipping vendored / build / VCS / worktree directories) and
reports every occurrence of TODO / FIXME / HACK as a *comment marker* —
explicitly requiring a `//` (C++/CUDA/HIP/GLSL) or `#` (Python) comment
prefix on the same line, so descriptive prose that happens to mention
"naive" / "placeholder" / "simplified" / "for now" does not trip the
linter.

Exit code:
    0  no NEW matches (anything already in the baseline is grandfathered in)
    1  one or more NEW matches printed to stderr

Use --strict to ignore the baseline and fail on every marker; use
--update-baseline after intentional changes to refresh the pinned set
(mirrors the pattern in tools/lint_test_contract.py and tools/audit_tolerances.py).

X.11: gated as a required CI check; the baseline lets pre-existing markers
in benchmarks / examples / archived tests stay quiet while still catching
any new TODO/HACK/etc. introduced by a PR.

CC.15 (audit-6): regex tightened from the original "any TODO/FIXME/HACK
*or* one of the soft phrases anywhere on the line" form. The soft phrases
("for now" / "placeholder" / "simplified" / "naive") fired on perfectly
fine descriptive comments — e.g. "A naive memcpy past these strides would
…" — producing 21 false positives in audit-5 against zero real new
markers. The TODO/FIXME/HACK keywords now require a comment-introducer
prefix (`//` or `#`) on the same line, matching how the project actually
spells real action-required markers.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "tools" / "scan_audit_markers_baseline.txt"

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

# Files whose basename matches any entry here are skipped regardless of the
# directory they live in. These are genuinely self-referential tools: they
# mention TODO/FIXME/HACK as *data* (a regex to match, a skip-reason category
# to count, or the marker literals they themselves search for) rather than as
# action markers, so scanning them would always produce false positives on a
# clean tree. Keep this set tiny and only add genuinely self-referential tools.
EXCLUDED_FILENAMES = {
    "scan_audit_markers.py",   # defines the marker literals it searches for
    "audit_tolerances.py",     # JUSTIFY_RX matches FIXME|TODO as justification text
    "count_skips.py",          # counts and reports "TODO:" skip-reason messages
}

# CC.15: only flag TODO/FIXME/HACK when they appear inside a comment on the
# same line. We match a `//` (C-family) or `#` (Python) introducer that
# precedes the keyword anywhere on the line. The keyword check is
# case-sensitive — lowercase `todo`/`hack` in prose ("the hack would be…")
# is not a marker; project convention is to write the keyword in all caps
# when it's meant as an action item.
MARKER_PATTERN = re.compile(
    r"(?://|#).*?\b(?:TODO|FIXME|HACK)\b"
)


def iter_source_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for fname in filenames:
            if fname in EXCLUDED_FILENAMES:
                continue
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


def _load_baseline(path: Path) -> set[str]:
    """Parse a baseline file of one `path:line` token per line.

    Same shape as tools/lint_test_contract_baseline.txt — comments start with
    `#` and blank lines are ignored. The baseline pins pre-existing markers
    so the lint passes if every hit is grandfathered in, and fails on any
    NEW marker.
    """
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def _token(rel_path: Path, lineno: int) -> str:
    """Canonical baseline token: `<repo-relative path>:<lineno>`."""
    return f"{rel_path.as_posix()}:{lineno}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Baseline file of known-existing markers (one `path:line` per "
        "line).  Pre-existing entries are ignored; new markers fail the lint.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Ignore the baseline; fail on every marker.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline file with the current set of markers.",
    )
    args = parser.parse_args()

    hits: list[tuple[Path, int, str]] = []
    for path in iter_source_files(REPO_ROOT):
        for lineno, line in scan_file(path):
            rel = path.relative_to(REPO_ROOT)
            hits.append((rel, lineno, line.strip()))

    if args.update_baseline:
        tokens = sorted({_token(rel, lineno) for rel, lineno, _ in hits})
        args.baseline.write_text(
            "# Baseline of known audit-marker hits (TODO/FIXME/HACK/...).\n"
            "# Each line is `<repo-relative path>:<lineno>` matching the\n"
            "# scanner's output. Pre-existing entries are ignored; new\n"
            "# markers FAIL the lint. Drive this list down over time.\n"
            + "\n".join(tokens)
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {len(tokens)} entries -> {args.baseline}")
        return 0

    baseline = set() if args.strict else _load_baseline(args.baseline)

    new_hits = [(rel, lineno, line) for rel, lineno, line in hits
                if _token(rel, lineno) not in baseline]
    grandfathered = len(hits) - len(new_hits)

    if new_hits:
        for rel, lineno, line in new_hits:
            print(f"{rel}:{lineno}: {line}")
        print(
            f"\nfound {len(new_hits)} NEW audit-marker hit(s) "
            f"(plus {grandfathered} pre-existing in baseline). "
            f"Either remove the markers or, if the addition is intentional, "
            f"regenerate the baseline with "
            f"`python3 tools/scan_audit_markers.py --update-baseline`.",
            file=sys.stderr,
        )
        return 1

    if grandfathered:
        print(
            f"OK: {grandfathered} pre-existing markers (all in baseline), "
            f"0 new markers.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
