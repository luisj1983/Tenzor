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


def _env_flag(name):
    """Match tests/backend_parity/golden_util.hpp's env_flag(): a var is
    "on" if it's set, non-empty, and doesn't start with '0'."""
    v = os.getenv(name, "")
    return bool(v) and not v.startswith("0")


def _require_multi_backend():
    """JIT-R129: single Python-side source of truth for the C++ suite's
    TENZOR_REQUIRE_MULTI_BACKEND convention -- turns an unavailable-backend
    skip into a hard failure so a CI box that's supposed to have e.g. ROCm
    but where the driver silently failed to init doesn't just quietly skip
    that parametrization."""
    return _env_flag("TENZOR_REQUIRE_MULTI_BACKEND")
ALL_DEVICES = [
    d for d in ["cpu", "cuda", "vulkan", "oneapi", "rocm", "mps"]
    if d not in _SKIP_BACKENDS
]

# CPU + any GPU backends actually present on the host. Handy when you want
# to skip a test entirely unless there's at least one GPU.
AVAILABLE_DEVICES = (["cpu"] if "cpu" not in _SKIP_BACKENDS else []) + [
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

    unavailable_reason = None
    if dev == "cuda" and not tz.cuda_is_available():
        unavailable_reason = "CUDA not available"
    elif dev == "vulkan" and not tz.vulkan_is_available():
        unavailable_reason = "Vulkan not available"
    elif dev == "oneapi" and not tz.oneapi_is_available():
        unavailable_reason = "OneAPI not available"
    elif dev == "rocm" and not tz.rocm_is_available():
        unavailable_reason = "ROCm not available"
    elif dev == "mps" and not tz.mps_is_available():
        unavailable_reason = "MPS not available"

    if unavailable_reason is not None:
        # JIT-R129: honor TENZOR_REQUIRE_MULTI_BACKEND the same way the C++
        # suite does (tests/backend_parity/golden_util.hpp's require_multi_
        # backend()) -- a bare skip here would silently hide a CI box that's
        # supposed to have this backend but where the driver failed to init.
        if _require_multi_backend():
            pytest.fail(
                f"Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1) "
                f"but {dev} is unavailable: {unavailable_reason}"
            )
        pytest.skip(unavailable_reason)

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


# FINDING 9: python/bindings/bindings_distributed.cpp's init_process_group()
# defaults its backend arg to "nccl" (a real, Python-exposed backend), but no
# Python test ever constructed a process group with it before this fixture —
# the `pg` fixture above always hardcodes "gloo". Skips (rather than fails)
# when neither CUDA nor ROCm is available, since NCCL/RCCL are GPU-only.
#
# Caveat: on a host with BOTH CUDA and ROCm present, ProcessGroup::
# create_process_group()'s Backend::NCCL case (src/distributed/distributed.cpp)
# resolves the active GPU backend via the registry and tries CUDA first,
# falling back to ROCm only when no CUDA backend is registered — so this
# fixture exercises whichever one wins that lookup (CUDA on a dual-GPU box),
# not both. Backend-specific isolation (as done at the C++ layer in
# tests/distributed/test_nccl_backend_smoke_rocm.cpp, a separate executable
# linking only tenzor_backend_rocm) isn't practical for a shared pytest
# process without forking, since NCCLBackend is a distinct compiled type per
# backend DSO.
@pytest.fixture
def pg_nccl():
    """Single-rank NCCL/RCCL process group; skips if no CUDA or ROCm GPU."""
    if not (tz.cuda_is_available() or tz.rocm_is_available()):
        pytest.skip("NCCL/RCCL requires CUDA or ROCm; neither is available")
    try:
        tz.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    except Exception as exc:
        pytest.skip(f"init_process_group(backend='nccl') unavailable: {exc}")
    yield tz.distributed.get_process_group()
    try:
        tz.distributed.destroy_process_group()
    except Exception:
        pass


def gpu_device():
    """First available GPU device (CUDA preferred, else ROCm), or None."""
    if tz.cuda_is_available():
        return tz.Device.cuda(0)
    if tz.rocm_is_available():
        return tz.Device.rocm(0)
    return None
