"""
Fifth-pass audit regression coverage for python/torch_interop.cpp.

B1: UInt16/UInt32/UInt64 must round-trip between PyTorch and Tenzor.
    Pre-fix, dtype_to_torch / dtype_from_torch only handled UInt8 and
    threw on the wider unsigned dtypes (which PyTorch added in 2.3).
"""
from __future__ import annotations

import sys

import pytest

# Skip the whole module if PyTorch is unavailable.
torch = pytest.importorskip("torch")

# Older PyTorch versions don't have torch.uint16/uint32/uint64.
HAS_WIDE_UINT = all(hasattr(torch, name) for name in ("uint16", "uint32", "uint64"))

sys.path.insert(0, "python")
import tenzor as tz  # noqa: E402

tz.initialize()


@pytest.mark.skipif(not HAS_WIDE_UINT, reason="PyTorch < 2.3 lacks wide unsigned dtypes")
@pytest.mark.parametrize(
    "torch_dtype,tz_dtype_name",
    [
        (torch.uint16, "uint16"),
        (torch.uint32, "uint32"),
        (torch.uint64, "uint64"),
    ],
)
def test_uint_dtypes_round_trip_torch_to_tenzor(torch_dtype, tz_dtype_name):
    t = torch.zeros(8, dtype=torch_dtype)
    # Should not throw "Unsupported PyTorch ScalarType for Tenzor".
    z = tz.from_torch(t)
    assert z.dtype == getattr(tz.dtype, tz_dtype_name)
    assert z.numel() == 8


@pytest.mark.skipif(not HAS_WIDE_UINT, reason="PyTorch < 2.3 lacks wide unsigned dtypes")
@pytest.mark.parametrize(
    "tz_dtype_name,torch_dtype",
    [
        ("uint16", torch.uint16),
        ("uint32", torch.uint32),
        ("uint64", torch.uint64),
    ],
)
def test_uint_dtypes_round_trip_tenzor_to_torch(tz_dtype_name, torch_dtype):
    z = tz.zeros((8,), dtype=getattr(tz.dtype, tz_dtype_name))
    # Should not throw "Unsupported DType for PyTorch conversion".
    t = tz.to_torch(z)
    assert t.dtype == torch_dtype
    assert t.numel() == 8
