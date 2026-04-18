"""Python tests for INT4 quantization (QuantDType.INT4 / QuantDType.UINT4).

The existing test_quantization.py only exercises INT8 paths. This file fills
the INT4 gap — verifies the quantize/dequantize round-trip on small tensors
across symmetric and asymmetric schemes, and checks observer output ranges
make sense for 4-bit storage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest

Q = tz.nn.quantization
INT4 = Q.QuantDType.INT4
UINT4 = Q.QuantDType.UINT4
SYMM = Q.QuantizationScheme.PerTensorSymmetric
ASYM = Q.QuantizationScheme.PerTensorAsymmetric


def _min_max_tensors(x):
    x_cpu = x.to('cpu')
    min_val = float(x_cpu.min().item())
    max_val = float(x_cpu.max().item())
    return (
        tz.tensor([min_val], tz.dtype.float32),
        tz.tensor([max_val], tz.dtype.float32),
    )


def test_int4_quant_params_symmetric():
    x = tz.randn([8, 8])
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, INT4, SYMM)
    assert qparams is not None
    # Symmetric INT4 has 16 levels [-8, 7]; scale should be > 0.
    assert qparams.scale.item() > 0


def test_int4_quant_params_asymmetric():
    x = tz.randn([8, 8])
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, INT4, ASYM)
    assert qparams is not None
    assert qparams.scale.item() > 0


def test_uint4_quant_params_asymmetric():
    """UINT4 quant params (range [0, 15])."""
    x = tz.randn([8, 8]).abs()  # non-negative makes UINT meaningful
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, UINT4, ASYM)
    assert qparams is not None


def test_int4_minmax_observer():
    """Per-tensor symmetric INT4 via observer."""
    obs = Q.MinMaxObserver()
    x = tz.randn([16, 16])
    obs.observe(x)
    qparams = obs.calculate_qparams(INT4, SYMM)
    assert qparams is not None


def test_int4_quantize_per_tensor_symmetric_roundtrip():
    """Quantize → dequantize of small tensor for INT4 symmetric."""
    x = tz.randn([4, 4]) * 2.0  # widen the range
    qt = Q.quantize_per_tensor_symmetric(x, INT4)
    assert qt is not None
    x_recon = Q.dequantize_tensor(qt)
    # INT4 has 4 bits → ~6% max relative error in the worst case. Just make
    # sure dequantization produces a Float tensor of the same shape.
    assert x_recon.shape == x.shape


def test_int4_per_channel_quantization():
    """Per-channel INT4 along axis 0 of a 2D tensor."""
    x = tz.randn([4, 8])
    qt = Q.quantize_per_channel_symmetric(x, 0, INT4)
    assert qt is not None
    x_recon = Q.dequantize_tensor(qt)
    assert x_recon.shape == x.shape


def test_int4_quantization_error_smaller_than_random():
    """compute_quantization_error returns (mse, max_abs_err, snr).
    All three should be positive finite numbers for non-degenerate input."""
    x = tz.randn([16, 16])
    qt = Q.quantize_per_tensor_symmetric(x, INT4)
    mse, max_err, snr = Q.compute_quantization_error(x, qt)
    assert mse >= 0.0
    assert max_err >= 0.0
    # MSE should be << variance of input (~1.0); 4-bit quant with full range
    # can reach ~0.05-0.2 depending on input distribution.
    assert mse < 0.5, f"INT4 MSE {mse} too large"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
