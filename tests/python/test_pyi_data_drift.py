"""Regression test for LL.12: DataLoader.__init__ parameter order drift.

The stub ``python/tenzor/data.pyi`` previously listed
``DataLoader.__init__`` parameters in a different order than the runtime
class in ``python/tenzor/data.py``, and the default for ``timeout`` was
``0`` in the stub but ``60.0`` at runtime. Static analysis tools that
honour the stub (mypy, pyright, IDEs) silently mis-interpreted positional
calls.

This test introspects the runtime ``DataLoader.__init__`` signature and
asserts that the parameter names appear in the same order, with the same
defaults, in ``data.pyi``. It uses ``ast`` to parse the stub rather than
``inspect`` because pyi modules are not importable at runtime.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import tenzor.data as tdata
from tenzor.data import DataLoader

PYI_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "python"
    / "tenzor"
    / "data.pyi"
)


def _runtime_params() -> list[inspect.Parameter]:
    sig = inspect.signature(DataLoader.__init__)
    # Drop ``self``.
    return [p for name, p in sig.parameters.items() if name != "self"]


def _stub_dataloader_init() -> ast.FunctionDef:
    tree = ast.parse(PYI_PATH.read_text(encoding="utf-8"), filename=str(PYI_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DataLoader":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
    raise AssertionError("DataLoader.__init__ not found in data.pyi")


def _stub_params() -> list[tuple[str, ast.expr | None]]:
    """Return (name, default-AST) pairs in declared order, excluding ``self``."""
    fn = _stub_dataloader_init()
    args = fn.args
    # Positional + keyword-or-positional args. We don't expect kw-only here.
    names = [a.arg for a in args.args if a.arg != "self"]
    # Defaults align with the *trailing* len(defaults) args.
    defaults: list[ast.expr | None] = [None] * (len(names) - len(args.defaults))
    defaults.extend(args.defaults)
    return list(zip(names, defaults))


def _ast_constant_value(node: ast.expr | None):
    if node is None:
        return inspect.Parameter.empty
    if isinstance(node, ast.Constant):
        return node.value
    # Negative numerics show up as UnaryOp(USub, Constant(...)).
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    # Stub may use ``None`` literal for Optional defaults — already handled by Constant.
    return ast.unparse(node)


def test_dataloader_pyi_param_order_matches_runtime():
    runtime = _runtime_params()
    stub = _stub_params()
    runtime_names = [p.name for p in runtime]
    stub_names = [n for n, _ in stub]
    assert stub_names == runtime_names, (
        "DataLoader.__init__ parameter order drift between runtime "
        f"({runtime_names}) and stub ({stub_names})"
    )


def test_dataloader_pyi_defaults_match_runtime():
    runtime = _runtime_params()
    stub = _stub_params()
    for rp, (sname, sdefault_node) in zip(runtime, stub):
        sdefault = _ast_constant_value(sdefault_node)
        if rp.default is inspect.Parameter.empty:
            assert sdefault is inspect.Parameter.empty, (
                f"{rp.name}: stub has default {sdefault!r} but runtime has none"
            )
            continue
        # Skip non-trivial runtime defaults (callables/objects); the stub
        # may use ``None`` or a sentinel — we only enforce constants.
        if not isinstance(rp.default, (int, float, str, bool, type(None))):
            continue
        assert sdefault == rp.default, (
            f"{rp.name}: stub default {sdefault!r} != runtime default {rp.default!r}"
        )
