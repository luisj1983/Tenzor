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
    """Allocate a leaf tensor that participates in autograd."""
    g = tz.Generator()
    g.manual_seed(seed)
    t = tz.randn(shape, generator=g)
    if hasattr(t, "requires_grad_"):
        t.requires_grad_(True)
    return t


def assign_grad(tz, param, seed):
    """Synthesise a deterministic gradient for the parameter."""
    g = tz.Generator()
    g.manual_seed(seed)
    grad = tz.randn(param.shape, generator=g)
    # Tenzor tensors expose .grad as an assignable attribute.
    param.grad = grad


def clone_param(tz, param):
    """Detach-clone a parameter for an independent optimiser instance."""
    cloned = param.detach().clone() if hasattr(param, "detach") else param.clone()
    if hasattr(cloned, "requires_grad_"):
        cloned.requires_grad_(True)
    return cloned


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
    """Convert a tenzor tensor to a list of floats for value comparison."""
    cpu = t.to("cpu") if hasattr(t, "to") else t
    if hasattr(cpu, "tolist"):
        return cpu.tolist()
    if hasattr(cpu, "numpy"):
        return cpu.numpy().tolist()
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


def check_optimizer(tz, name, cls) -> bool:
    print(f"[check] {name}")
    # Two independent parameters → one per param-group.
    p1_a = make_param(tz, (4, 4), seed=1)
    p2_a = make_param(tz, (4,), seed=2)
    p1_b = clone_param(tz, p1_a)
    p2_b = clone_param(tz, p2_a)

    # Construct a 2-group instance. Most optimisers require `lr`; pass a
    # safe value that all of them accept. Any class-specific defaults are
    # respected per group.
    try:
        ref = cls([
            {"params": [p1_a]},
            {"params": [p2_a]},
        ], lr=1e-2)
    except TypeError:
        ref = cls([
            {"params": [p1_a]},
            {"params": [p2_a]},
        ])

    # Step once so the optimiser populates its internal state buffers
    # (momentum, exp_avg, …) — pickling an unstepped state is uninteresting.
    assign_grad(tz, p1_a, seed=10)
    assign_grad(tz, p2_a, seed=11)
    ref.step()

    # Serialise the post-first-step state to a tempfile.
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
            other = cls([
                {"params": [p1_b]},
                {"params": [p2_b]},
            ], lr=1e-2)
        except TypeError:
            other = cls([
                {"params": [p1_b]},
                {"params": [p2_b]},
            ])

        with open(tmp_path, "rb") as fh:
            loaded_sd = pickle.load(fh)
        try:
            other.load_state_dict(loaded_sd)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  FAIL: load_state_dict() raised: {exc!r}")
            return False

        # Apply the *same* matched step to ref by stepping once on a copy of
        # the post-load parameters: we already advanced p?_a by one step,
        # but p?_b has not yet advanced. Apply the same first-step grads
        # to p?_b so both sides have one optimiser step's worth of update
        # baked in, then continue for `NUM_STEPS` more steps in lockstep.
        assign_grad(tz, p1_b, seed=10)
        assign_grad(tz, p2_b, seed=11)
        other.step()

        diff_p1 = max_abs_diff(to_numpy_like(tz, p1_a), to_numpy_like(tz, p1_b))
        diff_p2 = max_abs_diff(to_numpy_like(tz, p2_a), to_numpy_like(tz, p2_b))
        if max(diff_p1, diff_p2) > ABS_TOL:
            print(
                f"  FAIL: post-load step diverged after warm-up "
                f"(p1 Δ={diff_p1:.3e}, p2 Δ={diff_p2:.3e})"
            )
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
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import tenzor as tz  # type: ignore
        import tenzor.optim as optim  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        print(f"FATAL: cannot import tenzor.optim: {exc!r}")
        traceback.print_exc()
        return 1
    if hasattr(tz, "initialize"):
        try:
            tz.initialize()
        except Exception:  # pylint: disable=broad-except
            pass  # already initialised

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
    sys.exit(main())
