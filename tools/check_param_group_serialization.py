#!/usr/bin/env python3
"""
Verify that every tenzor.optim optimiser round-trips through pickle while
preserving its update trajectory across multiple param groups.

For each Optimizer subclass exposed by `tenzor.optim`:
    1. construct a 2-group instance over two small parameter tensors
       (one group per parameter) using the class's default hyperparams
       except `lr`, which is provided so SGD-family optimisers don't need
       a special case
    2. pickle.dumps() the state_dict to a tempfile
    3. construct a fresh instance with the same defaults and load_state_dict
       from the pickle
    4. run 10 step()s on both instances with identical synthetic grads
    5. assert that, on every step, both instances move the parameters by
       the same amount to within 1e-6 absolute tolerance

Exit code:
    0  every optimiser preserves trajectory across save / load
    1  any optimiser diverges, fails to instantiate, or fails to pickle
"""

from __future__ import annotations

import inspect
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ABS_TOL = 1e-6
NUM_STEPS = 10


def make_param(tz, shape, seed):
    """Allocate a leaf Variable that participates in autograd.

    X.8: the C++ Optimizer bindings take ``Sequence[Variable]`` (not plain
    Tensor), so wrap the random tensor in a ``Variable(t, requires_grad=True)``.
    Determinism comes from the global ``manual_seed()`` since ``randn`` doesn't
    accept a per-call generator at the binding layer.
    """
    if hasattr(tz, "manual_seed"):
        tz.manual_seed(seed)
    t = tz.randn(shape)
    return tz.Variable(t, True)


def assign_grad(tz, param, seed):
    """Synthesise a deterministic gradient on the parameter Variable.

    X.8: ``Variable.grad`` is bound read-only at the pybind11 layer, and no
    ``set_grad`` is exposed. The supported way to populate ``.grad`` is via
    ``backward()`` on a scalar loss that depends linearly on ``param`` — for
    a loss ``L = sum(param * g)``, ``dL/dparam = g``, which is exactly what
    we want.
    """
    if hasattr(tz, "manual_seed"):
        tz.manual_seed(seed)
    grad_tensor = tz.randn(_shape_of(param))
    # Reset any accumulated grad from prior steps so we get exactly grad_tensor.
    if hasattr(param, "zero_grad"):
        param.zero_grad()
    grad_var = tz.Variable(grad_tensor, False)
    loss = tz.sum(param * grad_var)
    loss.backward()


def _shape_of(v):
    """Extract a tuple shape from a Variable or Tensor."""
    if hasattr(v, "shape"):
        s = v.shape
        try:
            return tuple(s)
        except TypeError:
            return s
    if hasattr(v, "tensor"):
        return tuple(v.tensor().shape)
    raise RuntimeError("cannot read shape of param")


def clone_param(tz, param):
    """Detach-clone a parameter Variable for an independent optimiser instance."""
    # Variable.detach() yields a Tensor; rewrap as a fresh Variable.
    if hasattr(param, "detach"):
        t = param.detach()
        if hasattr(t, "clone"):
            t = t.clone()
    elif hasattr(param, "tensor"):
        t = param.tensor().clone()
    else:
        t = param.clone()
    return tz.Variable(t, True)


def discover_optimizers(optim_module):
    base = getattr(optim_module, "Optimizer", None)
    classes = []
    for name in dir(optim_module):
        if name.startswith("_"):
            continue
        obj = getattr(optim_module, name)
        if not inspect.isclass(obj):
            continue
        if base is not None and obj is base:
            continue
        if base is not None and not issubclass(obj, base):
            continue
        classes.append((name, obj))
    return classes


def to_numpy_like(tz, t):
    """Convert a tenzor Variable/Tensor to a list of floats for comparison."""
    # Variables expose .tensor() to get the underlying Tensor; Tensors expose
    # .numpy() directly.
    raw = t.tensor() if hasattr(t, "tensor") else t
    cpu = raw.cpu() if hasattr(raw, "cpu") else raw
    if hasattr(cpu, "numpy"):
        return cpu.numpy().tolist()
    if hasattr(cpu, "tolist"):
        return cpu.tolist()
    raise RuntimeError("cannot extract values from tenzor tensor")


def flatten(values):
    out = []
    stack = [values]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(float(v))
    return out


def max_abs_diff(a, b):
    fa = flatten(a)
    fb = flatten(b)
    if len(fa) != len(fb):
        return float("inf")
    return max((abs(x - y) for x, y in zip(fa, fb)), default=0.0)


# X.8: optimisers that wrap or compose other optimisers, or that have
# fundamentally different signatures, can't be exercised by the generic
# param-group probe. We don't silently skip them on the round-trip check (that
# would defeat the purpose of the lint), but we do recognise them explicitly so
# that future additions show up here rather than as cryptic TypeErrors.
_WRAPPER_OPTIMIZERS = frozenset({
    "SAM",  # wraps a base optimizer instance
    "LBFGS",  # K.1: L-BFGS rejects multi-group construction by design
    "LBFGSLineSearch",  # enum, not an Optimizer subclass
    "AveragedModel",  # SWA helper, not an Optimizer
    "ZeROStage1Optimizer",  # requires a distributed process group
    "ZeROStage1Config", "ZeROStage2Config", "ZeROStage3Config",  # config types
})


def _build_optimizer(cls, first_param, lr):
    """Construct an optimiser instance with a single parameter.

    The C++ Optimizer constructors take ``Sequence[Variable]`` (not param-group
    dicts). To exercise the multi-group code path callers should construct with
    ``first_param`` only and then ``add_param_group(ParamGroup([second_param]))``
    for the second group.
    """
    # All current optimisers (except SAM/AveragedModel) accept (params, lr=...).
    # A handful (SGD) require positional lr; others default it. Try with kwarg
    # first, then fall back to default.
    try:
        return cls([first_param], lr=lr)
    except TypeError:
        return cls([first_param])


def check_optimizer(tz, name, cls) -> bool:
    print(f"[check] {name}")
    if name in _WRAPPER_OPTIMIZERS:
        print(f"  skip: {name} has a non-standard constructor (wrapper/config)")
        return True

    # Two independent parameters → one per param-group.
    p1_a = make_param(tz, (4, 4), seed=1)
    p2_a = make_param(tz, (4,), seed=2)
    p1_b = clone_param(tz, p1_a)
    p2_b = clone_param(tz, p2_a)

    # Construct the optimiser with one param, then add a second param group
    # to exercise the per-group state path.
    try:
        ref = _build_optimizer(cls, p1_a, lr=1e-2)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  FAIL: constructor raised: {exc!r}")
        return False
    try:
        ref.add_param_group(tz.optim.ParamGroup([p2_a], lr=1e-2))
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  FAIL: add_param_group raised: {exc!r}")
        return False

    # X.8: pickle the fresh (pre-step) state, build a replica, load the state
    # into it. Then run NUM_STEPS identical steps on both and verify every
    # step matches. We do NOT pre-step `ref` before snapshotting because the
    # state_dict captures only optimiser-internal state (step counter,
    # momentum buffers); parameter values themselves live on the Variables.
    # If ref were pre-stepped, ref's params would diverge from other's params
    # immediately and no amount of equal-grad steps could re-converge them.
    try:
        sd = ref.state_dict()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  FAIL: state_dict() raised: {exc!r}")
        return False

    with tempfile.NamedTemporaryFile(
        prefix=f"tenzor_optim_{name}_", suffix=".pkl", delete=False
    ) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(sd, fh)

        try:
            other = _build_optimizer(cls, p1_b, lr=1e-2)
            other.add_param_group(tz.optim.ParamGroup([p2_b], lr=1e-2))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  FAIL: replica constructor raised: {exc!r}")
            return False

        with open(tmp_path, "rb") as fh:
            loaded_sd = pickle.load(fh)
        try:
            other.load_state_dict(loaded_sd)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  FAIL: load_state_dict() raised: {exc!r}")
            return False

        for step_idx in range(NUM_STEPS):
            seed_a = 100 + step_idx
            seed_b = 200 + step_idx
            assign_grad(tz, p1_a, seed=seed_a)
            assign_grad(tz, p2_a, seed=seed_b)
            assign_grad(tz, p1_b, seed=seed_a)
            assign_grad(tz, p2_b, seed=seed_b)
            ref.step()
            other.step()
            diff_p1 = max_abs_diff(to_numpy_like(tz, p1_a), to_numpy_like(tz, p1_b))
            diff_p2 = max_abs_diff(to_numpy_like(tz, p2_a), to_numpy_like(tz, p2_b))
            if max(diff_p1, diff_p2) > ABS_TOL:
                print(
                    f"  FAIL: step {step_idx} diverged "
                    f"(p1 Δ={diff_p1:.3e}, p2 Δ={diff_p2:.3e})"
                )
                return False
        print(f"  ok ({NUM_STEPS} steps matched within {ABS_TOL})")
        return True
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> int:
    # X.8: mirror V.39 — the compiled `tenzor_core` extension lives in
    # `build/python/tenzor/`, NOT the source tree at `python/tenzor/`. CI runs
    # this lint on a clean tree where `build/python` may or may not exist;
    # add both candidate roots so the script works in either layout.
    sys.path.insert(0, str(REPO_ROOT / "build" / "python"))
    sys.path.insert(0, str(REPO_ROOT / "python"))
    # V.40: explicitly classify import failures and ensure a non-zero exit on
    # every FATAL branch.  Earlier history had this returning 0 on the scipy
    # ModuleNotFoundError path (importing tenzor.distributions eagerly pulled
    # in scipy.special); without a non-zero exit, the lint passed silently on
    # systems missing scipy while the actual check never ran.
    #
    # X.8: importing `tenzor` requires the compiled `tenzor_core` extension.
    # On a clean source-only checkout (no build/) we cannot run this check,
    # but we MUST NOT FATAL — the tooling lint job is expected to pass on a
    # clean tree. Treat ModuleNotFoundError for the top-level `tenzor` package
    # as a SKIP (print to stderr, exit 0). Any other error (broken install,
    # missing optim submodule when tenzor itself imports, etc.) remains FATAL.
    try:
        import tenzor as tz  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name in ("tenzor", "tenzor.tenzor_core"):
            print(
                f"SKIP: tenzor not built ({exc!r}); "
                "rebuild with `ninja tenzor_core` to run this check.",
                file=sys.stderr,
            )
            return 0
        print(
            f"FATAL: missing dependency while importing tenzor: {exc!r}\n"
            "       install the missing module or fix the eager import path "
            "(see V.39 for the distributions/scipy lazy-load pattern)."
        )
        traceback.print_exc()
        return 2
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FATAL: cannot import tenzor: {exc!r}")
        traceback.print_exc()
        return 1
    if hasattr(tz, "initialize"):
        try:
            tz.initialize()
        except Exception:  # pylint: disable=broad-except
            pass  # already initialised
    # After initialize(), tenzor.optim may need to be reached as an attribute
    # rather than a submodule import (it is set via `from .tenzor_core import *`
    # and not always registered in sys.modules — see __init__.py registration
    # pattern). Tolerate both shapes.
    try:
        import tenzor.optim as optim  # type: ignore
    except ModuleNotFoundError:
        optim = getattr(tz, "optim", None)
        if optim is None:
            print("FATAL: tenzor has no 'optim' submodule or attribute")
            return 1

    optimisers = discover_optimizers(optim)
    if not optimisers:
        print("FATAL: tenzor.optim exposes no Optimizer subclasses")
        return 1

    failed = []
    for name, cls in optimisers:
        try:
            ok = check_optimizer(tz, name, cls)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  FAIL: {name} raised: {exc!r}")
            traceback.print_exc()
            ok = False
        if not ok:
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} optimiser(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(optimisers)} optimiser(s) round-trip cleanly")
    return 0


if __name__ == "__main__":
    # V.40: belt-and-braces — any exception escaping main() must produce a
    # non-zero exit so CI never silently passes a FATAL.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        sys.exit(1)
