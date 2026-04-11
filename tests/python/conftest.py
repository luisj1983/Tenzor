"""Shared pytest fixtures and helpers for the Tenzor test suite."""

import sys
import os
import pytest

# numpy is optional at the conftest level — individual tests that need it
# should use ``pytest.importorskip("numpy")``. Making this import
# unconditional would block pytest collection on environments where numpy
# isn't installed even though many of the tenzor tests don't touch it.
try:
    import numpy as np
    _HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# Ensure the build directory's Python package is on the path
_build_python = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.isdir(_build_python):
    sys.path.insert(0, os.path.abspath(_build_python))

import tenzor as tz

# Initialize once per session
_initialized = False

def _ensure_init():
    global _initialized
    if not _initialized:
        tz.initialize()
        _initialized = True

_ensure_init()


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device(request):
    """Parametrize over available devices.

    Usage::

        @pytest.mark.parametrize("device", ["cpu", "cuda", "vulkan", "oneapi", "rocm"], indirect=True)
        def test_foo(device):
            x = tz.randn([3], device=device)
    """
    dev = request.param
    if dev == "cuda" and not tz.cuda_is_available():
        pytest.skip("CUDA not available")
    elif dev == "vulkan" and not tz.vulkan_is_available():
        pytest.skip("Vulkan not available")
    elif dev == "oneapi" and not tz.oneapi_is_available():
        pytest.skip("OneAPI not available")
    elif dev == "rocm" and not tz.rocm_is_available():
        pytest.skip("ROCm not available")
    return dev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_tensors_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    """Assert two tensors are numerically close."""
    a_np = np.array(a.to("cpu").to(tz.dtype.float32).contiguous().numpy()
                    if hasattr(a, 'numpy') else a)
    b_np = np.array(b.to("cpu").to(tz.dtype.float32).contiguous().numpy()
                    if hasattr(b, 'numpy') else b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol, err_msg=msg)


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not tz.cuda_is_available():
        pytest.skip("CUDA not available")
