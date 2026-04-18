#!/usr/bin/env python3
"""Backend parity: every non-CPU backend must match CPU numerically.

Parameterized over all backends via the ``device`` fixture from
``conftest.py``; unavailable backends are skipped automatically (via the
fixture) and any listed in ``TENZOR_SKIP_BACKENDS`` are also skipped.
The CPU case is trivially skipped since there's no other side to compare.
"""

import pytest

import tenzor.tenzor_core as tz

from tolerances import (
    ELEMWISE_ATOL, ELEMWISE_RTOL,
    MATMUL_ATOL, MATMUL_RTOL,
    NN_ATOL, NN_RTOL,
    REDUCTION_ATOL, REDUCTION_RTOL,
)

# All non-CPU backends we want to compare against CPU. CPU-vs-CPU is skipped
# inside _resolve_device below since it's trivially equal.
NON_CPU_BACKENDS = ["cuda", "vulkan", "oneapi", "rocm"]


def _resolve_device(device_name):
    """Turn a backend-name string into a tz.Device. Skips CPU (self-parity
    would be vacuous) and any backend that's unavailable (defence-in-depth —
    the ``device`` fixture already skips these)."""
    if device_name == "cpu":
        pytest.skip("parity compares CPU against GPU backends")
    ctor = {
        "cuda":   tz.Device.cuda,
        "vulkan": tz.Device.vulkan,
        "oneapi": tz.Device.oneapi,
        "rocm":   tz.Device.rocm,
    }[device_name]
    return ctor(0)


def _close(a_cpu, a_other, rtol=ELEMWISE_RTOL, atol=ELEMWISE_ATOL):
    """CPU and other-device results are numerically close (Tenzor-only)."""
    a = a_cpu.to(tz.Device.cpu()).to(tz.dtype.float32).contiguous()
    b = a_other.to(tz.Device.cpu()).to(tz.dtype.float32).contiguous()
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    diff = tz.abs(a - b)
    threshold = atol + rtol * tz.abs(b)
    excess = tz.max(diff - threshold)
    max_abs_diff = tz.max(diff)
    assert excess.item() <= 0, (
        f"Numerical mismatch: max abs diff = {max_abs_diff.item():.2e}, "
        f"rtol={rtol}, atol={atol}"
    )


# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_zeros_parity(device):
    dev = _resolve_device(device)
    a = tz.zeros([4, 4], tz.dtype.float32, tz.Device.cpu())
    b = tz.zeros([4, 4], tz.dtype.float32, dev)
    _close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_ones_parity(device):
    dev = _resolve_device(device)
    a = tz.ones([4, 4], tz.dtype.float32, tz.Device.cpu())
    b = tz.ones([4, 4], tz.dtype.float32, dev)
    _close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_randn_shape_parity(device):
    """randn uses different RNG per device, so only check shape."""
    dev = _resolve_device(device)
    a = tz.randn([8, 8], tz.dtype.float32, tz.Device.cpu())
    b = tz.randn([8, 8], tz.dtype.float32, dev)
    assert a.shape == b.shape


# ---------------------------------------------------------------------------
# Arithmetic ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_add_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x + y
    z_other = x.to(dev) + y.to(dev)
    _close(z_cpu, z_other)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_sub_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x - y
    z_other = x.to(dev) - y.to(dev)
    _close(z_cpu, z_other)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_mul_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32)
    z_cpu = x * y
    z_other = x.to(dev) * y.to(dev)
    _close(z_cpu, z_other)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_div_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    y = tz.randn([4, 4], tz.dtype.float32) + 1.0  # avoid div-by-zero
    z_cpu = x / y
    z_other = x.to(dev) / y.to(dev)
    _close(z_cpu, z_other)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_matmul_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 8], tz.dtype.float32)
    y = tz.randn([8, 4], tz.dtype.float32)
    z_cpu = tz.matmul(x, y)
    z_other = tz.matmul(x.to(dev), y.to(dev))
    _close(z_cpu, z_other, rtol=MATMUL_RTOL, atol=MATMUL_ATOL)


# ---------------------------------------------------------------------------
# Reduction ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_sum_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    s_cpu = tz.sum(x)
    s_other = tz.sum(x.to(dev))
    _close(s_cpu, s_other, rtol=REDUCTION_RTOL, atol=REDUCTION_ATOL)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_mean_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    m_cpu = tz.mean(x)
    m_other = tz.mean(x.to(dev))
    _close(m_cpu, m_other, rtol=REDUCTION_RTOL, atol=REDUCTION_ATOL)


# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_sigmoid_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    _close(tz.sigmoid(x), tz.sigmoid(x.to(dev)))


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_tanh_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    _close(tz.tanh(x), tz.tanh(x.to(dev)))


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_exp_parity(device):
    dev = _resolve_device(device)
    x = tz.randn([4, 4], tz.dtype.float32)
    _close(tz.exp(x), tz.exp(x.to(dev)))


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_log_parity(device):
    dev = _resolve_device(device)
    x = tz.abs(tz.randn([4, 4], tz.dtype.float32)) + 0.01  # positive values
    _close(tz.log(x), tz.log(x.to(dev)))


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_sqrt_parity(device):
    dev = _resolve_device(device)
    x = tz.abs(tz.randn([4, 4], tz.dtype.float32)) + 0.01
    _close(tz.sqrt(x), tz.sqrt(x.to(dev)))


# ---------------------------------------------------------------------------
# NN layer parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_linear_parity(device):
    dev = _resolve_device(device)
    linear = tz.nn.Linear(4, 2)
    x_cpu = tz.Variable(tz.randn([2, 4], tz.dtype.float32), False)
    y_cpu = linear(x_cpu)
    # Move the SAME module so weights are identical
    linear.to(dev)
    x_other = tz.Variable(x_cpu.data.to(dev), False)
    y_other = linear(x_other)
    _close(y_cpu.data, y_other.data, rtol=NN_RTOL, atol=NN_ATOL)
