"""End-to-end smoke test for the Python @tz.jit decorator.

Phase 13 wrap-up: confirms that
  - `import tenzor as tz` succeeds (no shadowing / import-error regressions).
  - `tz.jit` is the Python decorator (callable, not the C++ pybind module).
  - `tz.jit` invokes through to the C++ pipeline and produces correct values.
  - `tz.jit.show_graph` and `tz.jit.cache_stats` resolve through the Python
    wrapper and return text/dict outputs after a compile.
"""
from __future__ import annotations

import os
import sys
import numpy as np


def _try_import_tenzor():
    """Import tenzor with a clear failure message — historically the
    Python __init__.py shadowed `tz.jit` and crashed at import time."""
    try:
        import tenzor as tz
    except Exception as e:  # pragma: no cover — surface clearly.
        print(f"FAIL: import tenzor raised: {e!r}", file=sys.stderr)
        raise SystemExit(2)
    return tz


def main() -> int:
    tz = _try_import_tenzor()

    # 1. Importing must yield a callable `tz.jit`, not a pybind module.
    if not callable(tz.jit):
        print(f"FAIL: tz.jit is {type(tz.jit).__name__}, expected a function",
              file=sys.stderr)
        return 1
    if type(tz.jit).__name__ == "module":
        print("FAIL: tz.jit is a module (probably the C++ tenzor_core.jit "
              "shadowing the Python wrapper)", file=sys.stderr)
        return 1

    # 2. The decorator must compile and execute correctly on a trivial
    #    function. If TENZOR_USE_MLIR_JIT is OFF this raises a
    #    JitNotEnabledError which we treat as a clean skip.
    tz.initialize()

    from tenzor.jit import JitNotEnabledError
    try:
        @tz.jit
        def f(x):
            return x + x

        x = tz.Variable(tz.full([4], 1.5, tz.dtype.float32))
        out = f(x)
    except JitNotEnabledError:
        print("SKIP: MLIR JIT not enabled in this build")
        return 0

    # @tz.jit handles return a Variable; reach through to the underlying
    # Tensor for the numerical assertion.
    if hasattr(out, "tensor"):
        out_t = out.tensor() if callable(out.tensor) else out.tensor
    else:
        out_t = out
    arr = out_t.cpu().numpy() if hasattr(out_t, "cpu") else np.array(out_t)
    if not np.allclose(arr, 3.0):
        print(f"FAIL: expected 3.0, got {arr}", file=sys.stderr)
        return 1
    print(f"OK: @tz.jit f(x) = x + x produced {arr.tolist()}")

    # 3. show_graph / show_mlir / cache_stats resolve through the Python
    #    wrapper and accept a single fn argument (defaulting to the last
    #    invocation example).
    g = tz.jit.show_graph(f)
    if not isinstance(g, str):
        print(f"FAIL: show_graph returned {type(g).__name__}, expected str",
              file=sys.stderr)
        return 1
    print(f"OK: tz.jit.show_graph(f) returned {len(g)} chars")

    stats = tz.jit.cache_stats()
    if not isinstance(stats, dict):
        print(f"FAIL: cache_stats returned {type(stats).__name__}, expected "
              f"dict", file=sys.stderr)
        return 1
    if "misses" not in stats:
        print(f"FAIL: cache_stats missing 'misses' key: {stats}",
              file=sys.stderr)
        return 1
    print(f"OK: tz.jit.cache_stats = {stats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
