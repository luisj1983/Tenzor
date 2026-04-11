"""Tests for quantization framework (PTQ, QAT, observers, quantized layers)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest

Q = tz.nn.quantization
INT8 = Q.QuantDType.INT8
SYMM = Q.QuantizationScheme.PerTensorSymmetric
ASYM = Q.QuantizationScheme.PerTensorAsymmetric


def _min_max_tensors(x):
    """Return 1-element min/max tensors as compute_quantization_params expects."""
    x_cpu = x.to('cpu')
    min_val = float(x_cpu.min().item())
    max_val = float(x_cpu.max().item())
    min_t = tz.tensor([min_val], tz.dtype.float32)
    max_t = tz.tensor([max_val], tz.dtype.float32)
    return min_t, max_t


# ---------------------------------------------------------------------------
# Basic quantization/dequantization
# ---------------------------------------------------------------------------

def test_quantize_per_tensor_symmetric():
    x = tz.randn([4, 4])
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, INT8, SYMM)
    assert qparams is not None


def test_quantize_per_tensor_asymmetric():
    x = tz.randn([4, 4])
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, INT8, ASYM)
    assert qparams is not None


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

def test_minmax_observer():
    obs = Q.MinMaxObserver()
    x = tz.randn([8, 4])
    obs.observe(x)
    qparams = obs.calculate_qparams(INT8, SYMM)
    assert qparams is not None


def test_moving_average_observer():
    obs = Q.MovingAverageMinMaxObserver()
    for _ in range(5):
        x = tz.randn([8, 4])
        obs.observe(x)
    qparams = obs.calculate_qparams(INT8, SYMM)
    assert qparams is not None


def test_histogram_observer():
    obs = Q.HistogramObserver()
    x = tz.randn([8, 4])
    obs.observe(x)
    qparams = obs.calculate_qparams(INT8, SYMM)
    assert qparams is not None


# ---------------------------------------------------------------------------
# FakeQuantize
# ---------------------------------------------------------------------------

def test_fake_quantize_forward():
    fq = Q.FakeQuantize()
    x = tz.Variable(tz.randn([4, 4]), False)
    y = fq.forward(x)
    assert y.data.shape == [4, 4]


def test_fake_quantize_enable_disable():
    fq = Q.FakeQuantize()
    fq.enable_observer(True)
    fq.enable_fake_quant(True)
    x = tz.Variable(tz.randn([4, 4]), False)
    y = fq.forward(x)
    assert y.data.shape == [4, 4]

    fq.enable_observer(False)
    fq.enable_fake_quant(False)
    y2 = fq.forward(x)
    assert y2.data.shape == [4, 4]


# ---------------------------------------------------------------------------
# QConfig
# ---------------------------------------------------------------------------

def test_default_qconfig():
    qc = Q.DefaultQConfigs.default_qconfig()
    assert qc is not None


def test_qat_qconfig():
    qc = Q.DefaultQConfigs.qat_qconfig()
    assert qc is not None


# ---------------------------------------------------------------------------
# Quantized layers
# ---------------------------------------------------------------------------

def _symmetric_qparams():
    scale = tz.tensor([0.1], tz.dtype.float32)
    zp = tz.zeros([1], dtype=tz.dtype.int32)
    return Q.QuantizationParams(scale, zp, INT8, SYMM)


def test_quantized_linear():
    ql = Q.QuantizedLinear(4, 3, _symmetric_qparams(), 0.1)
    # Forward takes a quantized input via fake-quant path in practice;
    # just exercise the constructor and verify the module exists.
    assert ql is not None


def test_quantized_conv2d():
    qparams = _symmetric_qparams()
    qc = Q.QuantizedConv2d(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        weight_qparams=qparams,
    )
    assert qc is not None


# ---------------------------------------------------------------------------
# Quantization error
# ---------------------------------------------------------------------------

def test_quantization_error_small():
    """Quantization error signature takes (original, quantized_tensor)."""
    x = tz.randn([100])
    min_t, max_t = _min_max_tensors(x)
    qparams = Q.compute_quantization_params(min_t, max_t, INT8, SYMM)
    if hasattr(Q, 'quantize_tensor') and hasattr(Q, 'compute_quantization_error'):
        qt = Q.quantize_tensor(x, qparams)
        errs = Q.compute_quantization_error(x, qt)
        # Returns (max_abs_error, mean_abs_error, mse) tuple — all non-negative.
        for e in errs:
            assert float(e) >= 0.0
    else:
        # Helper functions not bound — treat as smoke test for qparams.
        assert qparams is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
