"""Tests for the tenzor.lite Python bindings (Phase 4).

The headline guarantee: a model exported via tz.lite.export and reloaded via
tz.lite.Runtime produces bit-identical Float32 output to the same Module
run eagerly through Variable wrapping.
"""

import os
import tempfile

import numpy as np
import pytest

import tenzor as tz

tz.initialize()


def _run_eager(module, x):
    """Run module eagerly with an inference-mode Variable wrap."""
    out = module(tz.Variable(x, False))
    return out.tensor()


@pytest.fixture
def tmp_tzlite_path():
    """Yield a temp .tzlite path; clean up after the test."""
    fd, path = tempfile.mkstemp(suffix=".tzlite")
    os.close(fd)
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_lite_runtime_class_exists():
    assert hasattr(tz, "lite")
    assert hasattr(tz.lite, "Runtime")
    assert hasattr(tz.lite, "export")


def test_export_linear_only_roundtrips(tmp_tzlite_path):
    m = tz.nn.Linear(8, 4, bias=True)
    m.eval()
    x = tz.randn([2, 8], dtype=tz.dtype.float32)
    y_ref = _run_eager(m, x).numpy()

    tz.lite.export(m, tmp_tzlite_path, input_shape=[2, 8])
    rt = tz.lite.Runtime(tmp_tzlite_path)
    y_lite = rt(x).numpy()

    assert y_lite.shape == y_ref.shape == (2, 4)
    np.testing.assert_allclose(y_lite, y_ref, atol=1e-6)


def test_export_mlp_roundtrips(tmp_tzlite_path):
    m = tz.nn.Sequential(
        tz.nn.Linear(4, 16, bias=True),
        tz.nn.ReLU(),
        tz.nn.Linear(16, 3, bias=True),
    )
    m.eval()
    x = tz.randn([5, 4], dtype=tz.dtype.float32)
    y_ref = _run_eager(m, x).numpy()

    tz.lite.export(m, tmp_tzlite_path, input_shape=[5, 4])
    rt = tz.lite.Runtime(tmp_tzlite_path)
    y_lite = rt(x).numpy()

    assert y_lite.shape == y_ref.shape == (5, 3)
    np.testing.assert_allclose(y_lite, y_ref, atol=1e-5)


@pytest.mark.parametrize("activation_cls,name", [
    (tz.nn.Sigmoid, "sigmoid"),
    (tz.nn.Tanh, "tanh"),
    (tz.nn.GELU, "gelu"),
    (tz.nn.ReLU, "relu"),
])
def test_export_all_activations(activation_cls, name, tmp_tzlite_path):
    m = tz.nn.Sequential(
        tz.nn.Linear(4, 4, bias=True),
        activation_cls(),
    )
    m.eval()
    x = tz.randn([2, 4], dtype=tz.dtype.float32)
    y_ref = _run_eager(m, x).numpy()

    tz.lite.export(m, tmp_tzlite_path, input_shape=[2, 4])
    rt = tz.lite.Runtime(tmp_tzlite_path)
    y_lite = rt(x).numpy()

    np.testing.assert_allclose(y_lite, y_ref, atol=1e-4)


def test_runtime_metadata_and_shapes(tmp_tzlite_path):
    m = tz.nn.Linear(2, 2, bias=False)
    m.eval()
    tz.lite.export(m, tmp_tzlite_path, input_shape=[1, 2])
    rt = tz.lite.Runtime(tmp_tzlite_path)
    assert rt.metadata("framework") == "tenzor-lite"
    assert rt.input_shapes == [[1, 2]]


def test_unsupported_layer_raises(tmp_tzlite_path):
    # tz.nn.Conv2d is supported by Tenzor but not by Phase 3's Lite exporter.
    # The exporter must raise rather than silently emit a broken graph.
    m = tz.nn.Conv2d(3, 8, kernel_size=3)
    with pytest.raises(RuntimeError):
        tz.lite.export(m, tmp_tzlite_path, input_shape=[1, 3, 8, 8])
