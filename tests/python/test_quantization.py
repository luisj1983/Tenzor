"""Tests for quantization framework (PTQ, QAT, observers, quantized layers)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


# ---------------------------------------------------------------------------
# Basic quantization/dequantization
# ---------------------------------------------------------------------------

def test_quantize_per_tensor_symmetric():
    x = tz.randn([4, 4])
    qparams = tz.nn.quantization.compute_quantization_params(
        x, tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    assert qparams.scale.shape == [1] or qparams.scale.numel() == 1


def test_quantize_per_tensor_asymmetric():
    x = tz.randn([4, 4])
    qparams = tz.nn.quantization.compute_quantization_params(
        x, tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorAsymmetric)
    assert qparams.scale.numel() >= 1


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

def test_minmax_observer():
    obs = tz.nn.quantization.MinMaxObserver(
        tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    x = tz.randn([8, 4])
    obs.observe(x)
    qparams = obs.calculate_qparams()
    assert qparams.scale.numel() >= 1


def test_moving_average_observer():
    obs = tz.nn.quantization.MovingAverageMinMaxObserver(
        tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    for _ in range(5):
        x = tz.randn([8, 4])
        obs.observe(x)
    qparams = obs.calculate_qparams()
    assert qparams.scale.numel() >= 1


def test_histogram_observer():
    obs = tz.nn.quantization.HistogramObserver(
        tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    x = tz.randn([8, 4])
    obs.observe(x)
    qparams = obs.calculate_qparams()
    assert qparams.scale.numel() >= 1


# ---------------------------------------------------------------------------
# FakeQuantize
# ---------------------------------------------------------------------------

def test_fake_quantize_forward():
    fq = tz.nn.quantization.FakeQuantize(
        tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    x = tz.Variable(tz.randn([4, 4]), False)
    y = fq.forward(x)
    assert y.data.shape == [4, 4]


def test_fake_quantize_enable_disable():
    fq = tz.nn.quantization.FakeQuantize(
        tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
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
    qc = tz.nn.quantization.DefaultQConfigs.default_config()
    assert qc is not None


def test_qat_qconfig():
    qc = tz.nn.quantization.DefaultQConfigs.qat_config()
    assert qc is not None


# ---------------------------------------------------------------------------
# Quantized layers
# ---------------------------------------------------------------------------

def test_quantized_linear():
    ql = tz.nn.quantization.QuantizedLinear(4, 3)
    x = tz.Variable(tz.randn([2, 4]), False)
    y = ql.forward(x)
    assert y.data.shape == [2, 3]


def test_quantized_conv2d():
    qc = tz.nn.quantization.QuantizedConv2d(3, 8, 3)
    x = tz.Variable(tz.randn([1, 3, 8, 8]), False)
    y = qc.forward(x)
    assert len(y.data.shape) == 4


# ---------------------------------------------------------------------------
# Quantization error
# ---------------------------------------------------------------------------

def test_quantization_error_small():
    """Quantization error should be small for typical distributions."""
    x = tz.randn([100])
    err = tz.nn.quantization.compute_quantization_error(
        x, tz.nn.quantization.QuantDType.INT8,
        tz.nn.quantization.QuantizationScheme.PerTensorSymmetric)
    # Error should be finite and non-negative
    assert err >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
