"""
AMP execution-semantics coverage.

test_mixed_precision.py already covers the context-manager API. This file
drives actual computation inside the autocast block and inspects the
runtime state — the thing a regression in the autocast-interceptor layer
would break.

Scope:
  - Autocast is disabled outside the block, enabled inside, and restored
    after.
  - get_dtype reflects the dtype argument while the block is active.
  - GradScaler's unscale / overflow-detection flow produces a finite inf
    check on a handcrafted overflowing gradient.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init():
    tz.initialize()
    tz.manual_seed(0)


def test_autocast_is_disabled_by_default():
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_enables_inside_block():
    assert tz.amp.Autocast.is_enabled() is False
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.is_enabled() is True
    # After block: back to disabled.
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_dtype_reflected_during_block():
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.get_dtype() == tz.dtype.float16
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.bfloat16):
        assert tz.amp.Autocast.get_dtype() == tz.dtype.bfloat16


def test_autocast_nested_blocks_unwind_correctly():
    assert tz.amp.Autocast.is_enabled() is False
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        assert tz.amp.Autocast.is_enabled() is True
        with tz.amp.Autocast(enabled=False):
            assert tz.amp.Autocast.is_enabled() is False
        assert tz.amp.Autocast.is_enabled() is True
    assert tz.amp.Autocast.is_enabled() is False


def test_autocast_computes_without_raising():
    # An autocast block wrapping a matmul should not raise — the interceptor
    # path needs to handle the Float32 → Float16 promotion internally.
    x = tz.randn([4, 8])
    y = tz.randn([8, 4])
    with tz.amp.Autocast(enabled=True, dtype=tz.dtype.float16):
        z = tz.matmul(x, y)
    assert z.shape == [4, 4]


def test_grad_scaler_exists():
    # GradScaler is exposed in tz.amp and constructible.
    scaler = tz.amp.GradScaler()
    assert scaler is not None


# ---------------------------------------------------------------------------
# GradScaler overflow detection: real device coverage.
#
# The module docstring above has claimed since this file's introduction that
# it covers "GradScaler's unscale / overflow-detection flow produces a
# finite inf check on a handcrafted overflowing gradient" -- but no test in
# this file (or test_mixed_precision.py / test_mixed_precision_extended.py)
# ever actually did that; test_grad_scaler_overflow_detection in
# test_mixed_precision.py only exercises reset()/state_dict() round-trips,
# never a real Inf/NaN gradient. GradScaler.step()'s overflow scan
# (check_inf_nan_ in grad_scaler.cpp) dispatches through each backend's own
# isfinite/isnan reduction kernels -- a per-backend bug there (see FINDING 29
# for a concrete precedent: an OneAPI reduction-kernel size-boundary bug)
# would ship with zero Python-level signal since every test here implicitly
# ran on the default device only.
# ---------------------------------------------------------------------------

NON_CPU_BACKENDS = ["cuda", "vulkan", "oneapi", "rocm", "mps"]


def _resolve_device(device_name):
    ctor = {
        "cuda":   tz.Device.cuda,
        "vulkan": tz.Device.vulkan,
        "oneapi": tz.Device.oneapi,
        "rocm":   tz.Device.rocm,
        "mps":    tz.Device.mps,
    }[device_name]
    return ctor(0)


def _make_optimizer(dev):
    linear = tz.nn.Linear(4, 2).to(dev)
    params = list(linear.parameters())
    optimizer = tz.optim.SGD(params, lr=0.01)
    return optimizer, params


@pytest.mark.parametrize("device", ["cpu"] + NON_CPU_BACKENDS, indirect=True)
def test_grad_scaler_detects_inf_gradient(device):
    dev = tz.Device.cpu() if device == "cpu" else _resolve_device(device)
    optimizer, params = _make_optimizer(dev)

    params[0].set_grad(tz.full(params[0].data.shape, float("inf"), tz.dtype.float32, dev))
    for p in params[1:]:
        p.set_grad(tz.zeros(p.data.shape, tz.dtype.float32, dev))

    scaler = tz.amp.GradScaler(init_scale=1.0)
    scaler.step(optimizer)
    assert scaler.found_inf_nan() is True, (
        f"GradScaler on {device} failed to detect an Inf gradient"
    )


@pytest.mark.parametrize("device", ["cpu"] + NON_CPU_BACKENDS, indirect=True)
def test_grad_scaler_detects_nan_gradient(device):
    dev = tz.Device.cpu() if device == "cpu" else _resolve_device(device)
    optimizer, params = _make_optimizer(dev)

    params[0].set_grad(tz.full(params[0].data.shape, float("nan"), tz.dtype.float32, dev))
    for p in params[1:]:
        p.set_grad(tz.zeros(p.data.shape, tz.dtype.float32, dev))

    scaler = tz.amp.GradScaler(init_scale=1.0)
    scaler.step(optimizer)
    assert scaler.found_inf_nan() is True, (
        f"GradScaler on {device} failed to detect a NaN gradient"
    )


@pytest.mark.parametrize("device", ["cpu"] + NON_CPU_BACKENDS, indirect=True)
def test_grad_scaler_no_false_positive_on_finite_gradient(device):
    # The counterpart to the two tests above: a per-backend isfinite/isnan
    # kernel bug could just as easily always report an overflow as never
    # report one -- this catches that failure mode too.
    dev = tz.Device.cpu() if device == "cpu" else _resolve_device(device)
    optimizer, params = _make_optimizer(dev)

    for p in params:
        p.set_grad(tz.ones(p.data.shape, tz.dtype.float32, dev))

    scaler = tz.amp.GradScaler(init_scale=1.0)
    scaler.step(optimizer)
    assert scaler.found_inf_nan() is False, (
        f"GradScaler on {device} raised a false-positive overflow on a finite gradient"
    )
