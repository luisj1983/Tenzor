"""Regression tests for Module.to/cuda/cpu chainability (audit-9 LL.21).

KK.20 in audit-8 fixed the pybind11 bindings so ``Module::to(Device)``,
``Module::to(DType)``, ``Module::to(str)``, ``Module::cpu()`` and
``Module::cuda()`` all return ``self`` (via ``Module*``) instead of
``None``. This file regresses that contract so a future refactor that
drops the lambda wrappers fails CI instead of silently breaking user code
of the form ``model.cpu().to(tz.dtype.float32)``.
"""

import pytest

import tenzor as tz


def test_to_device_then_dtype_chain():
    """``m.to(device).to(dtype)`` must return a non-None module both times."""
    m = tz.nn.Linear(3, 5)
    ret = m.to(tz.Device(tz.DeviceType.CPU, 0)).to(tz.dtype.float32)
    assert ret is not None
    assert hasattr(ret, "to")  # chain returned something with .to
    # Parameters should be at the requested dtype/device
    for p in m.parameters():
        assert p.data.dtype == tz.dtype.float32
        # Variable's underlying Tensor exposes .device; Module.to(Device) moves params.
        assert p.data.device.type == tz.DeviceType.CPU


def test_to_string_chain():
    """``m.to("cpu")`` must also chain."""
    m = tz.nn.Linear(4, 2)
    ret = m.to("cpu")
    assert ret is not None
    # And chain another call onto it
    ret2 = ret.to(tz.dtype.float32)
    assert ret2 is not None
    assert hasattr(ret2, "parameters")


def test_cpu_chain():
    """``m.cpu()`` must chain."""
    m = tz.nn.Linear(4, 2)
    ret = m.cpu()
    assert ret is not None
    assert hasattr(ret, "to")
    # Chain a dtype conversion off the cpu() result
    ret2 = ret.to(tz.dtype.float32)
    assert ret2 is not None
    for p in m.parameters():
        assert p.data.dtype == tz.dtype.float32


@pytest.mark.cuda
def test_cuda_chain():
    """``m.cuda()`` must chain when CUDA is available."""
    if not tz.cuda_is_available():
        pytest.skip("CUDA not available")
    m = tz.nn.Linear(4, 2)
    ret = m.cuda()
    assert ret is not None
    assert hasattr(ret, "cpu")
    # Chain back to CPU off the cuda() result
    ret2 = ret.cpu()
    assert ret2 is not None
