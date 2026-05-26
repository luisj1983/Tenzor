"""
Tenzor Composable Function Transforms (torch.func equivalent)

Wraps the C++ ``tenzor_core.func`` submodule so the Python ``tenzor.func``
namespace has a corresponding ``.py`` source file (with ``func.pyi`` stub).

Provides ``grad``, ``vmap``, ``jacrev``, ``jacfwd``, ``hessian``, ``jvp``,
``hvp``, ``vhp`` and ``vjp`` — all backed by the C++ implementations
registered in ``python/bindings/bindings_autograd.cpp``.

Usage
-----
    import tenzor as tz

    f = lambda x: (x * x).sum()
    g = tz.func.grad(f)
    print(g(x))  # df/dx
"""

# audit-10 OO.12: expose the C++-side composable function transforms via a
# proper Python module so check_pyi_drift.py can diff func.pyi against this
# file (matches the autograd / nn / data pattern).
from . import tenzor_core as _core

_func_cpp = _core.func

grad = _func_cpp.grad
vmap = _func_cpp.vmap
jacrev = _func_cpp.jacrev
jacfwd = _func_cpp.jacfwd
hessian = _func_cpp.hessian
jvp = _func_cpp.jvp
hvp = _func_cpp.hvp
vhp = _func_cpp.vhp
vjp = _func_cpp.vjp

__all__ = [
    "grad",
    "hessian",
    "hvp",
    "jacfwd",
    "jacrev",
    "jvp",
    "vhp",
    "vjp",
    "vmap",
]
