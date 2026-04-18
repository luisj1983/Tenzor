"""Python tests for gradient compression bindings.

Distributed training compresses gradients before all-reduce to save bandwidth.
The two compressors bound to Python — FP16Compressor and TopKCompressor — had
no Python coverage. These tests verify compress → decompress preserves shape
and dtype, and that TopKCompressor's `ratio` argument controls how many
elements survive.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


def _shape_eq(a, b):
    return list(a.shape) == list(b.shape)


# ----------------------------------------------------------------------------
# FP16Compressor — casts gradient to fp16 for transport, restores to fp32.
# ----------------------------------------------------------------------------

def test_fp16_compressor_roundtrip_shape():
    comp = tz.distributed.FP16Compressor()
    grad = tz.randn([8, 8], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    restored = comp.decompress(compressed)
    assert _shape_eq(restored, grad), "FP16 round-trip must preserve shape"


def test_fp16_compressor_lossy_but_close():
    """FP16 round-trip introduces small rounding error but stays close."""
    comp = tz.distributed.FP16Compressor()
    grad = tz.randn([16, 16], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    restored = comp.decompress(compressed)
    diff = (restored - grad).abs().max().item()
    # FP16 has ~3 decimal digits of precision; max abs diff should be small.
    assert diff < 0.05, f"FP16 round-trip diff {diff} too large"


def test_fp16_compressor_compression_ratio():
    """FP16 stores half the bytes of FP32, so ratio should be ~0.5."""
    comp = tz.distributed.FP16Compressor()
    grad = tz.randn([32, 32], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    # Implementation defines compression_ratio as out_bytes / in_bytes.
    # For FP16 of an FP32 grad that's 0.5; allow a small tolerance for headers.
    assert 0.0 < compressed.compression_ratio <= 0.6


def test_fp16_compressor_name():
    comp = tz.distributed.FP16Compressor()
    name = comp.name()
    assert isinstance(name, str) and len(name) > 0


# ----------------------------------------------------------------------------
# TopKCompressor — keeps only the top-k% magnitude entries, zeros the rest.
# ----------------------------------------------------------------------------

def test_topk_compressor_roundtrip_shape_default_ratio():
    comp = tz.distributed.TopKCompressor()  # default ratio=0.01 (top 1%)
    grad = tz.randn([16, 16], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    restored = comp.decompress(compressed)
    assert _shape_eq(restored, grad)


def test_topk_compressor_custom_ratio_25pct():
    comp = tz.distributed.TopKCompressor(ratio=0.25)
    grad = tz.randn([16, 16], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    restored = comp.decompress(compressed)
    # ~75% of values should be zero after dequantization. Count via the tensor
    # API to avoid pulling in numpy.
    cpu = restored.to('cpu')
    zero_count = float((cpu == 0).to(tz.dtype.float32).sum().item())
    total = float(cpu.numel)
    zero_frac = zero_count / total
    assert 0.6 <= zero_frac <= 0.85, (
        f"TopKCompressor(ratio=0.25) should leave ~75% zeros; got {zero_frac:.2f}"
    )


def test_topk_compressor_compression_ratio_smaller_than_one():
    """TopK is sparser than dense storage → compression_ratio < 1."""
    comp = tz.distributed.TopKCompressor(ratio=0.05)
    grad = tz.randn([32, 32], dtype=tz.dtype.float32)
    compressed = comp.compress(grad)
    assert compressed.compression_ratio < 1.0


def test_topk_compressor_reset():
    """reset() must be callable and not raise."""
    comp = tz.distributed.TopKCompressor(ratio=0.1)
    grad = tz.randn([4, 4], dtype=tz.dtype.float32)
    comp.compress(grad)  # may accumulate state
    comp.reset()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
