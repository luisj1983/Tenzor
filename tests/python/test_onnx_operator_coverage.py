"""
ONNX operator-coverage matrix.

For every supported ONNX operator category, exports a small Sequential-wrapped
model that uses that operator, imports it back, and asserts the loaded handle
is non-None. The Tenzor ONNX tracer requires `Sequential` composition to
follow the module structure, so each test wraps its target op in Sequential.

The intent is to catch regressions where a Tenzor-ONNX mapping quietly breaks
for a specific operator without breaking the flagship model exports.
"""

import os
import sys
import tempfile

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init():
    tz.initialize()
    tz.manual_seed(0)


def _export_and_reimport(model, example_input, tag):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        tz.onnx.export(model, example_input, path)
        loaded = tz.onnx.load(path)
        assert loaded is not None, f"tz.onnx.load returned None for {tag}"
        return loaded
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _seq(*layers):
    m = tz.nn.Sequential(*layers)
    m.eval()
    return m


# Each test below exercises a distinct ONNX operator mapping.


def test_linear_matmul_gemm_export():
    _export_and_reimport(_seq(tz.nn.Linear(4, 2)), tz.randn([1, 4]), "Gemm")


def test_conv2d_export():
    _export_and_reimport(
        _seq(tz.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)),
        tz.randn([1, 3, 8, 8]),
        "Conv",
    )


def test_convtranspose2d_export():
    _export_and_reimport(
        _seq(tz.nn.ConvTranspose2d(3, 8, kernel_size=3, stride=1, padding=1)),
        tz.randn([1, 3, 8, 8]),
        "ConvTranspose",
    )


def test_batchnorm_export():
    # BatchNorm's running_mean/running_var buffers used to confuse the tracer
    # into throwing "Cannot automatically trace custom module structure".
    # The tracer now inspects named_buffers() for the running stats and
    # emits an ONNX BatchNormalization node directly.
    _export_and_reimport(_seq(tz.nn.BatchNorm2d(4)), tz.randn([1, 4, 8, 8]),
                         "BatchNormalization")


def test_layernorm_export():
    # LayerNorm has 1D weight+bias with no buffers; the tracer now routes it
    # through export_layernorm which emits an ONNX LayerNormalization node.
    _export_and_reimport(_seq(tz.nn.LayerNorm([8])), tz.randn([1, 8]),
                         "LayerNormalization")


# Pure activation modules don't own parameters, which the tracer currently
# refuses ("Cannot trace custom module without parameters"). Each activation
# test anchors a Linear before the activation so the sequential has at least
# one parameter bucket for the tracer to follow.


def test_relu_export():
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.ReLU()),
                         tz.randn([1, 8]), "Relu")


def test_sigmoid_export():
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.Sigmoid()),
                         tz.randn([1, 8]), "Sigmoid")


def test_tanh_export():
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.Tanh()),
                         tz.randn([1, 8]), "Tanh")


def test_gelu_export():
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.GELU()),
                         tz.randn([1, 8]), "Gelu")


def test_leaky_relu_export():
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.LeakyReLU(0.1)),
                         tz.randn([1, 8]), "LeakyRelu")


def test_maxpool_export():
    _export_and_reimport(
        _seq(tz.nn.Conv2d(3, 3, kernel_size=1),
             tz.nn.MaxPool2d(kernel_size=2, stride=2)),
        tz.randn([1, 3, 8, 8]), "MaxPool")


def test_avgpool_export():
    _export_and_reimport(
        _seq(tz.nn.Conv2d(3, 3, kernel_size=1),
             tz.nn.AvgPool2d(kernel_size=2, stride=2)),
        tz.randn([1, 3, 8, 8]), "AveragePool")


def test_dropout_export():
    # Dropout in eval mode is a passthrough, still tests the Dropout-node
    # mapping in the ONNX graph. Anchor with Linear for the tracer.
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.Dropout(0.5)),
                         tz.randn([1, 8]), "Dropout")


def test_sequential_composition_export():
    _export_and_reimport(
        _seq(tz.nn.Linear(4, 8), tz.nn.ReLU(), tz.nn.Linear(8, 2)),
        tz.randn([1, 4]),
        "Linear→ReLU→Linear",
    )


def test_identity_export():
    # Identity is a passthrough; anchor with Linear because Identity has no
    # parameters of its own (tracer limitation).
    _export_and_reimport(_seq(tz.nn.Linear(8, 8), tz.nn.Identity()),
                         tz.randn([1, 8]), "Identity")
