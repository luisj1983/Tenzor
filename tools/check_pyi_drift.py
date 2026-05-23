#!/usr/bin/env python3
"""
Check for drift between python/tenzor/*.py modules and their .pyi stubs.

For each .py module that has a sibling .pyi:
    1. import the .py module
    2. enumerate the public top-level names (functions, classes, vars that
       don't begin with '_') via `inspect`
    3. parse the .pyi via `ast` and collect the public names declared there
    4. diff the two sets and report MISSING (in .py but absent from .pyi)
       and EXTRA (in .pyi but not in the runtime module) entries

Exit code:
    0  no drift detected
    1  drift found (or a module failed to import / parse)
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_DIR = REPO_ROOT / "python" / "tenzor"


def public_runtime_names(module) -> set[str]:
    names: set[str] = set()
    # Prefer __all__ if explicitly declared — that's the module author's
    # source of truth. Otherwise fall back to enumerating dir() and filtering
    # to objects defined in this module (via inspect.getmodule).
    declared_all = getattr(module, "__all__", None)
    if isinstance(declared_all, (list, tuple)):
        names.update(str(n) for n in declared_all if not str(n).startswith("_"))
        return names
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if inspect.ismodule(obj):
            continue  # skip re-exported modules
        owner = inspect.getmodule(obj)
        if owner is not None and owner is not module:
            # Re-imported symbol from another module; don't require it in the
            # stub for this module specifically.
            continue
        names.add(name)
    return names


def public_pyi_names(pyi_path: Path) -> set[str]:
    src = pyi_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(pyi_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                names.add(node.target.id)
    return names


def check_module(py_path: Path) -> tuple[bool, list[str]]:
    """Return (had_drift, messages)."""
    pyi_path = py_path.with_suffix(".pyi")
    if not pyi_path.exists():
        # Nothing to diff against — not drift.
        return False, []
    mod_name = f"tenzor.{py_path.stem}"
    messages: list[str] = []
    try:
        module = importlib.import_module(mod_name)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(f"{mod_name}: IMPORT FAILED: {exc!r}")
        return True, messages
    try:
        pyi_names = public_pyi_names(pyi_path)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(f"{pyi_path}: PARSE FAILED: {exc!r}")
        return True, messages

    runtime_names = public_runtime_names(module)
    missing = sorted(runtime_names - pyi_names)
    extra = sorted(pyi_names - runtime_names)

    drift = False
    if missing:
        drift = True
        messages.append(
            f"{mod_name}: MISSING from {pyi_path.name}: {', '.join(missing)}"
        )
    if extra:
        drift = True
        messages.append(
            f"{mod_name}: EXTRA in {pyi_path.name} (no runtime symbol): "
            f"{', '.join(extra)}"
        )
    return drift, messages


def main() -> int:
    # Make the in-tree python/ package importable.
    sys.path.insert(0, str(REPO_ROOT / "python"))

    any_drift = False
    for py_path in sorted(PY_DIR.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        drifted, messages = check_module(py_path)
        for m in messages:
            print(m)
        any_drift = any_drift or drifted

    return 1 if any_drift else 0


if __name__ == "__main__":
    sys.exit(main())
