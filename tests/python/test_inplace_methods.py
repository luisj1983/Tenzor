"""Phase 2.1 — Python named in-place tensor methods.

Covers every in-place method exposed in python/bindings.cpp:
    add_, sub_, mul_, div_, pow_, neg_, reciprocal_, abs_, sqrt_, exp_,
    log_, sign_, relu_, sigmoid_, tanh_, clamp_, clamp_min_, clamp_max_,
    copy_, normal_, uniform_, fill_, zero_

Each test constructs a small tensor, runs the in-place method, and
asserts the value matches the expected out-of-place result.
"""

import math
import pytest

import tenzor as tz
import tenzor.tenzor_core as _core

f32 = _core.dtype.float32


def _flat(t):
    """Read a 1-D tensor as a Python list via .item() — no numpy needed."""
    return [t[i].item() for i in range(t.numel)]


def _allclose(a, b, tol=1e-5):
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def test_add_sub_mul_div_inplace():
    a = tz.ones((3,), dtype=f32)
    b = tz.full((3,), 2.0, dtype=f32)
    a.add_(b)
    assert _allclose(_flat(a), [3.0, 3.0, 3.0])
    a.sub_(b)
    assert _allclose(_flat(a), [1.0, 1.0, 1.0])
    a.mul_(b)
    assert _allclose(_flat(a), [2.0, 2.0, 2.0])
    a.div_(b)
    assert _allclose(_flat(a), [1.0, 1.0, 1.0])


def test_pow_inplace():
    t = tz.tensor([2.0, 3.0, 4.0])
    t.pow_(2.0)
    assert _allclose(_flat(t), [4.0, 9.0, 16.0])


def test_neg_inplace():
    t = tz.tensor([1.0, -2.0, 3.0, -4.0])
    t.neg_()
    assert _allclose(_flat(t), [-1.0, 2.0, -3.0, 4.0])


def test_reciprocal_inplace():
    t = tz.tensor([2.0, 4.0, 8.0])
    t.reciprocal_()
    assert _allclose(_flat(t), [0.5, 0.25, 0.125])


def test_sign_inplace():
    t = tz.tensor([-3.0, 0.0, 5.0])
    t.sign_()
    assert _allclose(_flat(t), [-1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Unary math
# ---------------------------------------------------------------------------

def test_abs_inplace():
    t = tz.tensor([1.0, -2.0, 3.0, -4.0])
    t.abs_()
    assert _allclose(_flat(t), [1.0, 2.0, 3.0, 4.0])


def test_sqrt_log_exp_inplace():
    t = tz.full((2,), 4.0, dtype=f32)
    t.sqrt_()
    assert _allclose(_flat(t), [2.0, 2.0])
    t.log_()
    assert _allclose(_flat(t), [math.log(2.0), math.log(2.0)])
    t.exp_()
    assert _allclose(_flat(t), [2.0, 2.0])


# ---------------------------------------------------------------------------
# Activations — these go through the real in-place dispatch kernels
# (OpId::ReLUInplace / SigmoidInplace / TanhInplace).
# ---------------------------------------------------------------------------

def test_relu_inplace():
    t = tz.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    t.relu_()
    assert _allclose(_flat(t), [0.0, 0.0, 0.0, 1.0, 2.0])


def test_sigmoid_inplace_at_zero():
    t = tz.zeros((1,), dtype=f32)
    t.sigmoid_()
    assert abs(t[0].item() - 0.5) < 1e-5


def test_tanh_inplace_at_zero():
    t = tz.zeros((1,), dtype=f32)
    t.tanh_()
    assert abs(t[0].item()) < 1e-5


# ---------------------------------------------------------------------------
# Clamps
# ---------------------------------------------------------------------------

def test_clamp_inplace():
    t = tz.tensor([-3.0, -1.0, 0.5, 2.0, 5.0])
    t.clamp_(-1.0, 1.0)
    assert _allclose(_flat(t), [-1.0, -1.0, 0.5, 1.0, 1.0])


def test_clamp_min_inplace():
    t = tz.tensor([-3.0, -1.0, 0.5, 2.0, 5.0])
    t.clamp_min_(0.0)
    assert _allclose(_flat(t), [0.0, 0.0, 0.5, 2.0, 5.0])


def test_clamp_max_inplace():
    t = tz.tensor([-3.0, -1.0, 0.5, 2.0, 5.0])
    t.clamp_max_(1.0)
    assert _allclose(_flat(t), [-3.0, -1.0, 0.5, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Copy / random fill — sanity rather than distributional precision.
# ---------------------------------------------------------------------------

def test_copy_inplace():
    dst = tz.zeros((3,), dtype=f32)
    src = tz.full((3,), 7.0, dtype=f32)
    dst.copy_(src)
    assert _allclose(_flat(dst), [7.0, 7.0, 7.0])


def test_copy_inplace_shape_mismatch_raises():
    dst = tz.zeros((3,), dtype=f32)
    src = tz.zeros((4,), dtype=f32)
    with pytest.raises(ValueError):
        dst.copy_(src)


def test_normal_inplace_rough_mean():
    t = tz.zeros((400,), dtype=f32)
    t.normal_(0.0, 1.0)
    vals = _flat(t)
    mean = sum(vals) / len(vals)
    # Generous bound — just catching completely wrong parameterization.
    assert abs(mean) < 0.3, f"normal_ mean too far from 0: {mean}"


def test_uniform_inplace_bounded():
    t = tz.zeros((200,), dtype=f32)
    t.uniform_(-2.0, 2.0)
    vals = _flat(t)
    assert min(vals) >= -2.0 - 1e-5
    assert max(vals) <= 2.0 + 1e-5


# ---------------------------------------------------------------------------
# Pre-existing methods (regression): fill_, zero_
# ---------------------------------------------------------------------------

def test_fill_and_zero_inplace():
    t = tz.zeros((4,), dtype=f32)
    t.fill_(9.0)
    assert _allclose(_flat(t), [9.0] * 4)
    t.zero_()
    assert _allclose(_flat(t), [0.0] * 4)
