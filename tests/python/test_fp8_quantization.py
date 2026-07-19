"""
FP8 quantization Python-surface coverage.

Tenzor exposes two FP8 formats (E4M3 and E5M2) plus a handful of helper
functions. These tests exercise:
  - the newly-bound FP8 dtype enums,
  - fp8_max_value / compute_fp8_scale arithmetic,
  - quantize_to_fp8 / dequantize_from_fp8 round-trip.

The dequantization path should reproduce input values within the known
precision of each FP8 format.
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


def test_fp8_dtypes_exposed():
    assert hasattr(tz.dtype, "fp8_e4m3")
    assert hasattr(tz.dtype, "fp8_e5m2")
    assert tz.dtype.fp8_e4m3 != tz.dtype.fp8_e5m2


def test_fp8_max_values():
    # E4M3 maxes out at 448 (standard IEEE-adjacent spec).
    assert tz.fp8_max_value(tz.dtype.fp8_e4m3) == pytest.approx(448.0)
    # E5M2 gets a larger range: 57344.
    assert tz.fp8_max_value(tz.dtype.fp8_e5m2) == pytest.approx(57344.0)


def test_fp8_scale_is_amax_over_fp8_max():
    # scale = amax / fp8_max
    assert tz.compute_fp8_scale(448.0, tz.dtype.fp8_e4m3) == pytest.approx(1.0)
    assert tz.compute_fp8_scale(224.0, tz.dtype.fp8_e4m3) == pytest.approx(0.5)
    assert tz.compute_fp8_scale(57344.0, tz.dtype.fp8_e5m2) == pytest.approx(1.0)


def test_fp8_quantize_dequantize_roundtrip_e4m3():
    # After the writer/reader fix (dtype.cpp): E4M3's max finite encoding is
    # exp=0xF, mantissa=0x6 = 448, not the previously-used exp=0xE/mantissa=0x7
    # (= 240). Constant-magnitude inputs now round-trip exactly at amax.
    x = tz.full([8], 2.0)
    fp8_tensor, params = tz.quantize_to_fp8(x, tz.dtype.fp8_e4m3)
    assert params.amax == pytest.approx(2.0, rel=1e-3)
    restored = tz.dequantize_from_fp8(fp8_tensor, params.scale)
    assert float(restored[0].item()) == pytest.approx(2.0, rel=1e-3)


def test_fp8_quantize_dequantize_roundtrip_e5m2():
    x = tz.full([8], 128.0)
    fp8_tensor, params = tz.quantize_to_fp8(x, tz.dtype.fp8_e5m2)
    assert params.amax == pytest.approx(128.0, rel=1e-3)
    restored = tz.dequantize_from_fp8(fp8_tensor, params.scale)
    # E5M2 (5 exp, 2 mantissa) has coarser spacing than E4M3 near the middle
    # of its range — one mantissa bit means worst-case 25% relative error.
    # Keep the tolerance loose for E5M2 specifically.
    assert float(restored[0].item()) == pytest.approx(128.0, rel=0.25)


# ---------------------------------------------------------------------------
# Cross-backend coverage.
#
# Every test above implicitly runs on the default device only (never
# CUDA/Vulkan/OneAPI/ROCm explicitly), despite real FP8 cast/quantize kernels
# existing on CUDA (src/backends/cuda/kernels/transform.cu, math.cu), Vulkan
# (src/backends/vulkan/kernels/cast_fp8e4m3_f32.comp, cast_fp8e5m2_f32.comp,
# cast_f32_fp8e4m3.comp and siblings), and OneAPI
# (oneapi_kernel_registry.cpp's explicit is_fp8() dispatch helpers) -- an
# FP8 cast/quantize bug specific to any of those kernels would ship with
# zero Python-level signal. Mirrors test_backend_parity.py's device/
# _resolve_device pattern (this file also imports the raw tenzor_core
# module rather than the wrapped tenzor package, so it needs its own copy
# rather than conftest.py's string-based ``device`` fixture).
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


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_fp8_quantize_dequantize_roundtrip_e4m3_device(device):
    dev = _resolve_device(device)
    x = tz.full([8], 2.0, tz.dtype.float32, dev)
    fp8_tensor, params = tz.quantize_to_fp8(x, tz.dtype.fp8_e4m3)
    assert params.amax == pytest.approx(2.0, rel=1e-3)
    restored = tz.dequantize_from_fp8(fp8_tensor, params.scale)
    assert float(restored[0].to(tz.Device.cpu()).item()) == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("device", NON_CPU_BACKENDS, indirect=True)
def test_fp8_quantize_dequantize_roundtrip_e5m2_device(device):
    dev = _resolve_device(device)
    x = tz.full([8], 128.0, tz.dtype.float32, dev)
    fp8_tensor, params = tz.quantize_to_fp8(x, tz.dtype.fp8_e5m2)
    assert params.amax == pytest.approx(128.0, rel=1e-3)
    restored = tz.dequantize_from_fp8(fp8_tensor, params.scale)
    assert float(restored[0].to(tz.Device.cpu()).item()) == pytest.approx(128.0, rel=0.25)
