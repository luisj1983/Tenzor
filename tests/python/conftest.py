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
# Shared device constants
# ---------------------------------------------------------------------------

# Canonical list of backends to parameterize tests over. Single source of
# truth — previously duplicated in test_multibackend_ops.py and a handful of
# other files. Tests import this via ``from conftest import ALL_DEVICES`` OR
# pull it in implicitly via ``@pytest.mark.parametrize("device", ALL_DEVICES,
# indirect=True)`` (the ``device`` fixture below handles the availability
# skip).
_SKIP_BACKENDS = {
    s.strip() for s in os.getenv("TENZOR_SKIP_BACKENDS", "").split(",") if s.strip()
}
ALL_DEVICES = [
    d for d in ["cpu", "cuda", "vulkan", "oneapi", "rocm", "mps"]
    if d not in _SKIP_BACKENDS
]

# CPU + any GPU backends actually present on the host. Handy when you want
# to skip a test entirely unless there's at least one GPU.
AVAILABLE_DEVICES = ["cpu"] + [
    name for name, check in (
        ("cuda",   lambda: tz.cuda_is_available()),
        ("vulkan", lambda: tz.vulkan_is_available()),
        ("oneapi", lambda: tz.oneapi_is_available()),
        ("rocm",   lambda: tz.rocm_is_available()),
        ("mps",    lambda: tz.mps_is_available()),
    ) if check() and name not in _SKIP_BACKENDS
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device(request):
    """Parametrize over available devices.

    Usage::

        @pytest.mark.parametrize("device", ALL_DEVICES, indirect=True)
        def test_foo(device):
            x = tz.randn([3], device=device)
    """
    dev = request.param
    if dev in _SKIP_BACKENDS:
        pytest.skip(f"{dev} excluded via TENZOR_SKIP_BACKENDS")
    if dev == "cuda" and not tz.cuda_is_available():
        pytest.skip("CUDA not available")
    elif dev == "vulkan" and not tz.vulkan_is_available():
        pytest.skip("Vulkan not available")
    elif dev == "oneapi" and not tz.oneapi_is_available():
        pytest.skip("OneAPI not available")
    elif dev == "rocm" and not tz.rocm_is_available():
        pytest.skip("ROCm not available")
    elif dev == "mps" and not tz.mps_is_available():
        pytest.skip("MPS not available")
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


# ---------------------------------------------------------------------------
# Distributed / process-group fixtures
# ---------------------------------------------------------------------------
# Consolidated here (was duplicated verbatim in test_collective_ops.py,
# test_tensor_parallel.py, and test_zero_optimizers.py). The single-rank
# gloo setup lets per-file tests exercise the distributed API surface
# without requiring MPI / multi-process infrastructure.
#
# The fixture skips only when gloo is genuinely unavailable in this build —
# missing RANK / WORLD_SIZE env vars are NOT a skip reason, since we pass
# rank=0 / world_size=1 explicitly.
@pytest.fixture
def pg():
    """Single-rank gloo process group shared across distributed tests."""
    try:
        tz.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception as exc:
        pytest.skip(f"init_process_group unavailable: {exc}")
    yield tz.distributed.get_process_group()
    try:
        tz.distributed.destroy_process_group()
    except Exception:
        pass
