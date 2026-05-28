#!/usr/bin/env python3
"""
S9: Verify the PyTorch-compatible signature of ``F.batch_norm``.

The runtime function previously had a Tenzor-specific signature
``(input, num_features, training, momentum, eps, weight, bias)`` which
disagreed with both PyTorch and the type stub in ``functional.pyi``.
S9 aligns it to PyTorch:

    F.batch_norm(input, running_mean=None, running_var=None,
                 weight=None, bias=None, training=True,
                 momentum=0.1, eps=1e-5)

These tests cover:

1. PyTorch-style call shape (running_mean/var keyword-only) does not
   raise TypeError.
2. Eval mode normalises by the provided running stats.
3. Training mode with no running stats normalises by batch stats only.
4. Training mode with running stats updates them in place.
5. Backward through batch_norm yields non-None grads on input/weight/bias.
"""

import math
import os
import sys

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz
import tenzor.tenzor_core as tzc
import tenzor.nn.functional as F


_INITIALIZED = False


def _init():
    global _INITIALIZED
    if not _INITIALIZED:
        tz.initialize()
        _INITIALIZED = True


def _full_tensor(shape, value, dtype=None):
    """Create a Float32 tensor filled with ``value``."""
    dt = dtype if dtype is not None else tzc.dtype.float32
    t = tzc.zeros(shape, dt)
    t.fill_(value)
    return t


def _ones_tensor(shape, dtype=None):
    dt = dtype if dtype is not None else tzc.dtype.float32
    return tzc.ones(shape, dt)


def _zeros_tensor(shape, dtype=None):
    dt = dtype if dtype is not None else tzc.dtype.float32
    return tzc.zeros(shape, dt)


def _allclose(a_t, b_t, atol=1e-5, rtol=1e-4):
    """Compare two Tensors via .numpy() + math.isclose-style assert."""
    a = a_t.numpy().reshape(-1)
    b = b_t.numpy().reshape(-1)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    for x, y in zip(a.tolist(), b.tolist()):
        diff = abs(x - y)
        tol = atol + rtol * abs(y)
        assert diff <= tol, (
            f"allclose failed: |{x} - {y}| = {diff} > "
            f"atol+rtol*|y| = {tol}"
        )


# ---------------------------------------------------------------------
# 1. PyTorch-style keyword call must not raise TypeError.
# ---------------------------------------------------------------------
def test_pytorch_style_kwargs_no_typeerror():
    _init()
    print("Testing PyTorch-style keyword call...")
    N, C, H, W = 2, 4, 3, 3
    x = tz.Variable(tzc.randn([N, C, H, W], tzc.dtype.float32), False)
    rm = _zeros_tensor([C])
    rv = _ones_tensor([C])
    w = tz.Variable(_ones_tensor([C]), False)
    b = tz.Variable(_zeros_tensor([C]), False)
    y = F.batch_norm(x, running_mean=rm, running_var=rv,
                     weight=w, bias=b, training=False)
    assert y.tensor().shape == [N, C, H, W], (
        f"Output shape wrong: {y.tensor().shape}")
    print("  PyTorch-style kwargs OK")


# ---------------------------------------------------------------------
# 2. Eval mode uses the provided running stats.
# ---------------------------------------------------------------------
def test_eval_mode_uses_running_stats():
    _init()
    print("Testing eval mode uses running stats...")
    # Use a fixed input so the eval-mode normalisation is deterministic:
    #   y = (x - rm) / sqrt(rv + eps) * weight + bias
    # With rm = 2.0, rv = 4.0, eps = 1e-5, weight = 1.0, bias = 0.0:
    #   y = (x - 2) / sqrt(4 + 1e-5) ≈ (x - 2) / 2.0000025
    N, C, H, W = 1, 2, 2, 2
    x_t = tzc.zeros([N, C, H, W], tzc.dtype.float32)
    x_t.fill_(6.0)  # All elements = 6.0 -> y ≈ (6-2)/2 = 2.0
    x = tz.Variable(x_t, False)
    rm = _full_tensor([C], 2.0)
    rv = _full_tensor([C], 4.0)
    eps = 1e-5
    y = F.batch_norm(x, running_mean=rm, running_var=rv,
                     weight=None, bias=None, training=False, eps=eps)
    expected_val = (6.0 - 2.0) / math.sqrt(4.0 + eps)
    expected = _full_tensor([N, C, H, W], expected_val)
    _allclose(y.tensor(), expected, atol=1e-5, rtol=1e-4)
    print("  eval mode normalisation OK (got val ≈ %.6f)" % expected_val)

    # With affine: weight=3, bias=1 -> y = (x - 2) / sqrt(4+eps) * 3 + 1
    w = tz.Variable(_full_tensor([C], 3.0), False)
    b = tz.Variable(_full_tensor([C], 1.0), False)
    y2 = F.batch_norm(x, running_mean=rm, running_var=rv,
                      weight=w, bias=b, training=False, eps=eps)
    expected_val2 = expected_val * 3.0 + 1.0
    expected2 = _full_tensor([N, C, H, W], expected_val2)
    _allclose(y2.tensor(), expected2, atol=1e-5, rtol=1e-4)
    print("  eval mode + affine OK (got val ≈ %.6f)" % expected_val2)


# ---------------------------------------------------------------------
# 3. Training mode with no running stats normalises by batch stats only.
# ---------------------------------------------------------------------
def test_training_no_running_stats_uses_batch_stats():
    _init()
    print("Testing training mode without running stats...")
    # Construct an input where batch mean is exactly the channel mean and
    # batch variance is non-zero, so the normalised output should have
    # mean ≈ 0 per channel.
    N, C, H, W = 4, 3, 2, 2
    x_t = tzc.randn([N, C, H, W], tzc.dtype.float32)
    x = tz.Variable(x_t, False)
    y = F.batch_norm(x, running_mean=None, running_var=None,
                     training=True)
    # The per-channel mean of y should be ~0.
    y_np = y.tensor().numpy()
    # Reduce over N, H, W (axes 0, 2, 3).
    per_chan_mean = y_np.mean(axis=(0, 2, 3))
    for c in range(C):
        assert abs(float(per_chan_mean[c])) < 1e-4, (
            f"channel {c} mean is {per_chan_mean[c]}, expected ~0"
        )
    print("  training mode no running stats OK "
          "(per-channel mean ≈ %.2e)" % float(abs(per_chan_mean).max()))


# ---------------------------------------------------------------------
# 4. Training mode with running stats updates them in place.
# ---------------------------------------------------------------------
def test_training_updates_running_stats_in_place():
    _init()
    print("Testing training mode updates running stats in place...")
    N, C, H, W = 2, 3, 2, 2
    x_t = tzc.randn([N, C, H, W], tzc.dtype.float32)
    x = tz.Variable(x_t, False)
    # Start running_mean = 0, running_var = 1 (PyTorch's BatchNorm defaults).
    rm = _zeros_tensor([C])
    rv = _ones_tensor([C])

    rm_before = rm.numpy().copy()
    rv_before = rv.numpy().copy()

    F.batch_norm(x, running_mean=rm, running_var=rv,
                 training=True, momentum=0.1)

    rm_after = rm.numpy()
    rv_after = rv.numpy()
    # At least one entry should have shifted away from its initial value
    # (unless the batch stats happen to be exactly 0 / 1, which is
    # astronomically unlikely with randn input).
    diff_rm = float(abs(rm_after - rm_before).max())
    diff_rv = float(abs(rv_after - rv_before).max())
    assert diff_rm > 1e-6, (
        f"running_mean was not updated: max diff = {diff_rm}")
    assert diff_rv > 1e-6, (
        f"running_var was not updated: max diff = {diff_rv}")
    print("  running stats updated in place "
          "(rm diff ≈ %.4f, rv diff ≈ %.4f)" % (diff_rm, diff_rv))


# ---------------------------------------------------------------------
# 5. Backward yields non-None gradients on input / weight / bias.
# ---------------------------------------------------------------------
def test_backward_flows_to_input_weight_bias():
    _init()
    print("Testing backward flow on F.batch_norm...")
    N, C, H, W = 2, 4, 3, 3
    x = tz.Variable(tzc.randn([N, C, H, W], tzc.dtype.float32), True)
    w = tz.Variable(_ones_tensor([C]), True)
    b = tz.Variable(_zeros_tensor([C]), True)

    y = F.batch_norm(x, running_mean=None, running_var=None,
                     weight=w, bias=b, training=True)
    # Reduce to scalar via mean.
    loss = tzc.mean(y)
    loss.backward()

    assert x.grad is not None, "x.grad is None — autograd graph severed"
    assert w.grad is not None, "w.grad is None — affine grad path broken"
    assert b.grad is not None, "b.grad is None — bias grad path broken"
    # And the grad tensors should have matching shapes.
    assert x.grad.shape == [N, C, H, W], (
        f"x.grad shape wrong: {x.grad.shape}")
    assert w.grad.shape == [C], f"w.grad shape wrong: {w.grad.shape}"
    assert b.grad.shape == [C], f"b.grad shape wrong: {b.grad.shape}"
    print("  backward OK (all three grads are populated)")


# ---------------------------------------------------------------------
# 6. Eval mode without running stats must raise (PyTorch contract).
# ---------------------------------------------------------------------
def test_eval_mode_without_stats_raises():
    _init()
    print("Testing eval mode without running stats raises...")
    x = tz.Variable(tzc.randn([1, 2, 2, 2], tzc.dtype.float32), False)
    raised = False
    try:
        F.batch_norm(x, running_mean=None, running_var=None,
                     training=False)
    except (ValueError, RuntimeError):
        raised = True
    assert raised, (
        "Expected ValueError/RuntimeError when training=False and no "
        "running stats provided, but call returned successfully."
    )
    print("  eval-without-stats correctly raises")


def main():
    try:
        test_pytorch_style_kwargs_no_typeerror()
        test_eval_mode_uses_running_stats()
        test_training_no_running_stats_uses_batch_stats()
        test_training_updates_running_stats_in_place()
        test_backward_flows_to_input_weight_bias()
        test_eval_mode_without_stats_raises()
        print("\nALL F.batch_norm tests passed.")
        return 0
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
