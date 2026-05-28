#!/usr/bin/env python3
"""Diff runtime-exposed Tenzor names vs. .pyi stub declarations.

This is the *parity* checker complementary to ``check_pyi_drift.py``.
``check_pyi_drift.py`` only diffs ``.py`` files that have a sibling ``.pyi``;
this script also covers the ``.pyi``-only stubs (e.g. ``__init__.pyi``,
``optim.pyi``, ``quantization.pyi``, ``fft.pyi``, ``sparse.pyi``,
``distributed.pyi``, ``compression.pyi``, ``profiler.pyi``,
``torch_interop.pyi``) by importing the corresponding runtime module
(``tenzor``, ``tenzor.optim``, ``tenzor.tenzor_core.fft`` for the C++
pybind11 submodules, etc.) and asserting every public name on the runtime
side is declared in the stub.

Exit code:
    0  no drift detected
    1  drift found (missing names in .pyi, or vapor names in .pyi)

Stub-side names tolerated as intentional via the in-file allowlists below
(documented inline). Hard-block fallbacks: every other gap is reported.

Usage::

    python tools/check_pyi_parity.py             # full report
    python tools/check_pyi_parity.py --quiet     # summary only

The script is wired into ``tests/python/test_pyi_drift.py``, which the
project's CTest harness runs as ``cpu_python_test_pyi_drift``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_DIR = REPO_ROOT / "python" / "tenzor"


# ---------------------------------------------------------------------------
# Mapping: .pyi stub -> runtime module to introspect.
#
# Several stubs live at ``tenzor/<name>.pyi`` but the runtime module is
# actually a pybind11 submodule of the C++ extension, exposed under
# ``tenzor.<name>`` via ``from .tenzor_core import *`` (see
# ``python/tenzor/__init__.py``). For those, ``importlib.import_module``
# of ``tenzor.<name>`` would fail because no ``tenzor/<name>.py`` exists
# for some of them — instead we walk to ``tenzor`` and pick up the
# submodule attribute, which is what user code sees.
# ---------------------------------------------------------------------------
STUB_TO_RUNTIME: dict[str, str] = {
    # ``tenzor/__init__.pyi`` covers the ``tenzor`` namespace itself.
    "__init__.pyi": "tenzor",
    # Pure-Python wrappers loaded by ``__init__.py``.
    "autograd.pyi": "tenzor.autograd",
    "data.pyi": "tenzor.data",
    "func.pyi": "tenzor.func",
    "functional.pyi": "tenzor.nn.functional",
    "jit.pyi": "tenzor.jit",
    "nn.pyi": "tenzor.nn",
    # C++ pybind11 submodules surfaced through ``tenzor_core``.
    "compression.pyi": "tenzor.tenzor_core.compression",
    "distributed.pyi": "tenzor.tenzor_core.distributed",
    "exceptions.pyi": "tenzor.exceptions",
    "fft.pyi": "tenzor.tenzor_core.fft",
    "optim.pyi": "tenzor.optim",
    "profiler.pyi": "tenzor.tenzor_core.profiler",
    "quantization.pyi": "tenzor.tenzor_core.quantization",
    "sparse.pyi": "tenzor.tenzor_core.sparse",
    # ``torch_interop.pyi`` is conditionally compiled — only present when
    # the C++ extension was built with TENZOR_HAS_TORCH. Handled below.
    "torch_interop.pyi": "tenzor.tenzor_core.torch_interop",
}


# Stubs that may be entirely absent from the runtime — they document an
# optional surface gated by a build flag. We only validate them if the
# runtime module is importable.
OPTIONAL_RUNTIME_STUBS: set[str] = {
    "torch_interop.pyi",
}


# Names that legitimately appear in __init__.pyi but are not on the
# ``tenzor`` runtime namespace (or vice versa). Each entry MUST include a
# justification comment. Adding to this list should be a last resort; the
# default is "fix the .pyi", not "allowlist".
#
# Format: (stub_filename, name) -> reason
ALLOWLIST: dict[tuple[str, str], str] = {
    # __init__.pyi declares typed-alias names that are not runtime symbols.
    ("__init__.pyi", "Shape"): "Type alias (Union[Tuple[int,...], List[int]])",
    ("__init__.pyi", "Scalar"): "Type alias (Union[int, float, bool])",
    ("__init__.pyi", "ArrayLike"): "Type alias used by Tensor constructors",
    # ``DType`` is the canonical typed name; runtime exposes it as the
    # ``dtype`` enum-like object. The two coexist deliberately.
    ("__init__.pyi", "DType"): "Stub-only alias for the runtime ``dtype`` enum",
    # Forward decls / convenience exception names that pybind registers
    # under PyExc_* hooks; pybind also exposes them on ``tenzor`` directly
    # so the wildcard import binds them — but the canonical typed surface
    # is ``tenzor.exceptions``.
    # (No extra entries needed; these names are already in the runtime.)
}


# ``from typing import …`` / ``from __future__ import …`` / ``from enum import …``
# imports are stub-only scaffolding (used for annotations) and never appear
# as runtime attributes of the corresponding module. Skip them so the
# "EXTRA in stub" diff doesn't drown in noise like ``Any, List, Optional``.
TYPING_IMPORT_SOURCES: frozenset[str] = frozenset({
    "typing",
    "typing_extensions",
    "__future__",
    "enum",
    "abc",
    "collections.abc",
    "dataclasses",
    "pathlib",
    "numpy",      # often imported `as np` for ndarray annotations
    "numpy.typing",
})


def public_pyi_names(pyi_path: Path) -> set[str]:
    """Extract module-level public names declared in a .pyi stub.

    Captures:
      - ``class Foo(...): ...``         -> ``Foo``
      - ``def foo(...) -> ...: ...``    -> ``foo``
      - ``foo: int``                    -> ``foo``
      - ``Bar = ...``                   -> ``Bar``
      - ``from .x import y as y``       -> ``y`` (explicit re-export)
      - ``from . import sub as sub``    -> ``sub``

    Skips imports from typing / __future__ / enum / abc / numpy etc. —
    those are stub-only scaffolding and never resolve to a runtime attribute.
    """
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
        elif isinstance(node, ast.ImportFrom):
            # PEP 484: a stub's ``from X import Y`` is annotation-use only.
            # Only ``from X import Y as Y`` (or ``from X import *``) marks
            # the name as a public re-export of THIS stub. Anything else is
            # type-checker scaffolding and must NOT be diffed against the
            # runtime module.
            module = node.module or ""
            if module in TYPING_IMPORT_SOURCES:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                # Explicit re-export: ``import Y as Y`` (same identifier).
                if alias.asname is not None and alias.asname == alias.name:
                    if not alias.asname.startswith("_"):
                        names.add(alias.asname)
                # Otherwise: annotation-use only; not a public name.
        elif isinstance(node, ast.Import):
            # Likewise for ``import X`` — only count as re-export if it's
            # ``import X as X``.
            for alias in node.names:
                if alias.name in TYPING_IMPORT_SOURCES:
                    continue
                if alias.asname is not None and alias.asname == alias.name:
                    if not alias.asname.startswith("_"):
                        names.add(alias.asname)
        elif isinstance(node, ast.If):
            # Walk into ``if TYPE_CHECKING:`` blocks — stubs there are still
            # visible to static checkers.
            for sub in node.body:
                if isinstance(
                    sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if not sub.name.startswith("_"):
                        names.add(sub.name)
                elif isinstance(sub, ast.AnnAssign) and isinstance(
                    sub.target, ast.Name
                ):
                    if not sub.target.id.startswith("_"):
                        names.add(sub.target.id)
                elif isinstance(sub, ast.Assign):
                    for tgt in sub.targets:
                        if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                            names.add(tgt.id)
    return names


def public_runtime_names(module) -> set[str]:
    """Public attribute names on a runtime module.

    For ``tenzor`` itself we want EVERY public attribute (the wildcard
    import from ``tenzor_core`` wired through __init__.py). For pybind11
    submodules we likewise enumerate dir().

    Filtering ``dir()``:
      - keep names whose owner is the inspected module or any sub-tenzor
        module (the wildcard-import case);
      - drop names whose owner is a stdlib/third-party module (math,
        threading, queue, dataclasses, numpy, …);
      - keep names with no detectable owner (pybind11 C-ext objects often
        report owner=None — these are the canonical case).

    Note: we deliberately do NOT prefer ``__all__`` here. A stub documents
    what static checkers see when users write ``tenzor.X`` — attribute-
    accessible names, not just ``from tenzor import *`` re-exports.
    """
    import inspect
    import typing

    names: set[str] = set()
    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue
        # Drop typing scaffolding leaked from the module body
        # (TypeVar, ``TYPE_CHECKING``, ``Protocol``, …).
        if isinstance(obj, typing.TypeVar):
            continue
        if obj is typing.TYPE_CHECKING:
            continue
        owner = inspect.getmodule(obj)
        if owner is not None and owner is not module:
            owner_name = getattr(owner, "__name__", "") or ""
            # Sub-tenzor modules (tenzor, tenzor.tenzor_core, tenzor.nn, …)
            # are intentional re-exports surfaced via wildcard imports;
            # count them as part of this module's public surface.
            if not (owner_name == "tenzor" or owner_name.startswith("tenzor.")):
                continue
        names.add(name)
    return names


def diff_module(
    stub_filename: str,
    pyi_path: Path,
    runtime_module_name: str,
) -> tuple[bool, list[str]]:
    """Diff one stub against its runtime. Return (had_drift, messages)."""
    messages: list[str] = []
    optional = stub_filename in OPTIONAL_RUNTIME_STUBS

    try:
        module = importlib.import_module(runtime_module_name)
    except Exception as exc:  # pylint: disable=broad-except
        if optional:
            messages.append(
                f"{stub_filename}: runtime module {runtime_module_name} "
                f"unavailable (optional, skipping): {type(exc).__name__}"
            )
            return False, messages
        messages.append(
            f"{stub_filename}: IMPORT FAILED for {runtime_module_name}: {exc!r}"
        )
        return True, messages

    try:
        pyi_names = public_pyi_names(pyi_path)
    except Exception as exc:  # pylint: disable=broad-except
        messages.append(f"{pyi_path.name}: PARSE FAILED: {exc!r}")
        return True, messages

    runtime_names = public_runtime_names(module)

    missing = sorted(n for n in (runtime_names - pyi_names))
    extra = sorted(n for n in (pyi_names - runtime_names))

    # Apply allowlist to extra (vapor) entries only — never to missing,
    # because hiding "missing" entries from the report would defeat the
    # purpose of the check.
    extra = [
        n for n in extra if (stub_filename, n) not in ALLOWLIST
    ]

    drift = False
    if missing:
        drift = True
        messages.append(
            f"{stub_filename}: MISSING from stub (in {runtime_module_name} "
            f"runtime, not declared in .pyi): {', '.join(missing)}"
        )
    if extra:
        drift = True
        messages.append(
            f"{stub_filename}: EXTRA in stub (declared in .pyi but absent "
            f"from {runtime_module_name} runtime): {', '.join(extra)}"
        )
    return drift, messages


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print drift summary; suppress per-stub messages.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Make the in-tree python/ package importable.
    sys.path.insert(0, str(REPO_ROOT / "python"))

    any_drift = False
    drift_count = 0
    checked = 0
    for stub_filename, runtime_module_name in STUB_TO_RUNTIME.items():
        pyi_path = PY_DIR / stub_filename
        if not pyi_path.exists():
            print(f"WARNING: stub {pyi_path} missing on disk; skipping")
            continue
        checked += 1
        drifted, messages = diff_module(
            stub_filename, pyi_path, runtime_module_name
        )
        if drifted:
            drift_count += 1
            any_drift = True
        if not args.quiet:
            for m in messages:
                print(m)

    if any_drift:
        print(
            f"\ncheck_pyi_parity: DRIFT in {drift_count}/{checked} stub(s). "
            f"See messages above."
        )
        return 1
    print(f"check_pyi_parity: OK — {checked} stub(s) match runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
