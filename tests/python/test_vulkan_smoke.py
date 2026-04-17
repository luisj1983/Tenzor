"""Vulkan backend smoke tests from Python.

The existing Python test suite (test_tensor_ops.py, test_autograd.py,
test_functional.py) doesn't exercise Vulkan. This file runs the minimum
representative ops on the Vulkan backend so a Python-only Tenzor user
would catch a Vulkan regression immediately instead of via C++ tests only.

Deliberately avoids numpy so the test runs on minimal environments.
"""

import os
import sys
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))
import tenzor as tz

tz.initialize()

pytestmark = pytest.mark.skipif(
    not tz.vulkan_is_available(), reason="Vulkan not available"
)


def _device():
    return tz.Device("vulkan:0")


def _all_close_scalar(tensor, expected, atol=1e-4):
    """Move a tensor to CPU and verify every element ≈ `expected`.

    Walks the tensor via numpy when available; falls back to flat
    __getitem__ otherwise. Tensor has no tolist() method so we can't
    rely on that path.
    """
    cpu = tensor.to(tz.Device("cpu")).contiguous()
    try:
        import numpy as np
        arr = np.array(cpu.numpy()).ravel()
        for v in arr:
            if not math.isclose(float(v), expected, abs_tol=atol):
                return False, float(v)
        return True, None
    except ImportError:
        pass
    # numpy-less fallback: scalar tensors expose item(); multi-element
    # tensors need flat indexing.
    shape = list(cpu.shape)
    if not shape:
        v = float(cpu.item())
        return math.isclose(v, expected, abs_tol=atol), v
    flat = cpu.reshape([-1])
    numel = int(flat.shape[0])
    for i in range(numel):
        v = float(flat[i].item())
        if not math.isclose(v, expected, abs_tol=atol):
            return False, v
    return True, None


class TestVulkanElementwise:
    def test_add(self):
        a = tz.ones([4, 4], device=_device())
        b = tz.ones([4, 4], device=_device())
        c = a + b
        assert str(c.device).startswith("vulkan")
        ok, bad = _all_close_scalar(c, 2.0)
        assert ok, f"add produced non-2.0 value: {bad}"

    def test_mul_scalar(self):
        a = tz.ones([8], device=_device())
        b = a * 3.0
        ok, bad = _all_close_scalar(b, 3.0)
        assert ok, f"mul produced non-3.0 value: {bad}"

    def test_matmul(self):
        a = tz.ones([4, 8], device=_device())
        b = tz.ones([8, 3], device=_device())
        c = tz.matmul(a, b)
        assert list(c.shape) == [4, 3]
        # Each output element = sum of 8 ones = 8.0
        ok, bad = _all_close_scalar(c, 8.0)
        assert ok, f"matmul produced non-8.0 value: {bad}"


class TestVulkanRoundTrip:
    def test_cpu_to_vulkan_to_cpu(self):
        """Move a known tensor cpu→vulkan, double on device, move back."""
        cpu = tz.Device("cpu")
        vk = _device()
        # Use a known-value tensor rather than random so the comparison is
        # dtype-precision-independent.
        x_cpu = tz.ones([16, 16], device=cpu) * 3.0
        x_vk = x_cpu.to(vk)
        y_vk = x_vk * 2.0
        y_cpu = y_vk.to(cpu)
        ok, bad = _all_close_scalar(y_cpu, 6.0)
        assert ok, f"roundtrip produced non-6.0 value: {bad}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
