#!/usr/bin/env python3
"""Backend parity tests: verify CPU vs CUDA produce numerically close results.

Uses pure Tenzor ops for comparison (no numpy dependency).
All tests are skipped if CUDA is not available.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz

# Default tolerances by operation category
ELEMWISE_RTOL = 1e-5
ELEMWISE_ATOL = 1e-5
MATMUL_RTOL = 1e-4
MATMUL_ATOL = 1e-4
REDUCTION_RTOL = 1e-5
REDUCTION_ATOL = 1e-5
NN_RTOL = 1e-4
NN_ATOL = 1e-4

HAS_CUDA = False


def _close(a_cpu, a_cuda, rtol=ELEMWISE_RTOL, atol=ELEMWISE_ATOL):
    """Check that CPU and CUDA results are numerically close using Tenzor ops."""
    a = a_cpu.to(tz.Device.cpu()).to(tz.dtype.float32).contiguous()
    b = a_cuda.to(tz.Device.cpu()).to(tz.dtype.float32).contiguous()

    a_shape = a.shape
    b_shape = b.shape
    assert a_shape == b_shape, f"Shape mismatch: {a_shape} vs {b_shape}"

    # Compute |a - b| <= atol + rtol * |b|  (element-wise)
    diff = tz.abs(a - b)
    threshold = atol + rtol * tz.abs(b)
    # max(diff - threshold) should be <= 0 for all elements
    excess = tz.max(diff - threshold)
    max_abs_diff = tz.max(diff)

    assert excess.item() <= 0, (
        f"Numerical mismatch: max abs diff = {max_abs_diff.item():.2e}, "
        f"rtol={rtol}, atol={atol}"
    )


# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------

def test_zeros_parity():
    a = tz.zeros([4, 4], tz.dtype.float32, tz.Device.cpu())
    b = tz.zeros([4, 4], tz.dtype.float32, tz.Device.cuda(0))
    _close(a, b, atol=0, rtol=0)


def test_ones_parity():
    a = tz.ones([4, 4], tz.dtype.float32, tz.Device.cpu())
    b = tz.ones([4, 4], tz.dtype.float32, tz.Device.cuda(0))
    _close(a, b, atol=0, rtol=0)


def test_randn_shape_parity():
    """randn uses different RNG per device, so only check shape."""
    a = tz.randn([8, 8], tz.dtype.float32, tz.Device.cpu())
    b = tz.randn([8, 8], tz.dtype.float32, tz.Device.cuda(0))
    assert a.shape == b.shape


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

def test_add_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x + y
    z_cuda = x.to(tz.Device.cuda(0)) + y.to(tz.Device.cuda(0))
    _close(z_cpu, z_cuda)


def test_sub_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x - y
    z_cuda = x.to(tz.Device.cuda(0)) - y.to(tz.Device.cuda(0))
    _close(z_cpu, z_cuda)


def test_mul_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x * y
    z_cuda = x.to(tz.Device.cuda(0)) * y.to(tz.Device.cuda(0))
    _close(z_cpu, z_cuda)


def test_div_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32) + 1.0  # avoid div-by-zero
    z_cpu = x / y
    z_cuda = x.to(tz.Device.cuda(0)) / y.to(tz.Device.cuda(0))
    _close(z_cpu, z_cuda)


def test_matmul_parity():
    x = tz.randn([4, 8], tz.dtype.float32)
    y = tz.randn([8, 4], tz.dtype.float32)
    z_cpu = tz.matmul(x, y)
    z_cuda = tz.matmul(x.to(tz.Device.cuda(0)), y.to(tz.Device.cuda(0)))
    _close(z_cpu, z_cuda, rtol=MATMUL_RTOL, atol=MATMUL_ATOL)


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

def test_sum_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    s_cpu = tz.sum(x)
    s_cuda = tz.sum(x.to(tz.Device.cuda(0)))
    _close(s_cpu, s_cuda, rtol=REDUCTION_RTOL, atol=REDUCTION_ATOL)


def test_mean_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    m_cpu = tz.mean(x)
    m_cuda = tz.mean(x.to(tz.Device.cuda(0)))
    _close(m_cpu, m_cuda, rtol=REDUCTION_RTOL, atol=REDUCTION_ATOL)


# ---------------------------------------------------------------------------
# Activation ops (using nn modules since functions aren't all top-level)
# ---------------------------------------------------------------------------

def test_sigmoid_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y_cpu = tz.sigmoid(x)
    y_cuda = tz.sigmoid(x.to(tz.Device.cuda(0)))
    _close(y_cpu, y_cuda)


def test_tanh_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y_cpu = tz.tanh(x)
    y_cuda = tz.tanh(x.to(tz.Device.cuda(0)))
    _close(y_cpu, y_cuda)


def test_exp_parity():
    x = tz.randn([4, 4], tz.dtype.float32)
    y_cpu = tz.exp(x)
    y_cuda = tz.exp(x.to(tz.Device.cuda(0)))
    _close(y_cpu, y_cuda)


def test_log_parity():
    x = tz.abs(tz.randn([4, 4], tz.dtype.float32)) + 0.01  # positive values
    y_cpu = tz.log(x)
    y_cuda = tz.log(x.to(tz.Device.cuda(0)))
    _close(y_cpu, y_cuda)


def test_sqrt_parity():
    x = tz.abs(tz.randn([4, 4], tz.dtype.float32)) + 0.01
    y_cpu = tz.sqrt(x)
    y_cuda = tz.sqrt(x.to(tz.Device.cuda(0)))
    _close(y_cpu, y_cuda)


# ---------------------------------------------------------------------------
# NN layer parity
# ---------------------------------------------------------------------------

def test_linear_parity():
    linear = tz.nn.Linear(4, 2)
    x_cpu = tz.Variable(tz.randn([2, 4], tz.dtype.float32), False)
    y_cpu = linear(x_cpu)

    # Move the SAME module to CUDA so weights are identical
    linear.to(tz.Device.cuda(0))
    x_cuda = tz.Variable(x_cpu.data.to(tz.Device.cuda(0)), False)
    y_cuda = linear(x_cuda)

    _close(y_cpu.data, y_cuda.data, rtol=NN_RTOL, atol=NN_ATOL)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tz.initialize()

    HAS_CUDA = tz.cuda_is_available()
    if not HAS_CUDA:
        print("CUDA not available — skipping all parity tests")
        sys.exit(0)

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
