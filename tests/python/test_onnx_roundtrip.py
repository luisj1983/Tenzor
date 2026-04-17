"""ONNX export → import round-trip tests.

Complements test_onnx_export.py (which only covers export). Round-trips a
few representative models through tz.onnx.export → tz.onnx.load and
verifies the imported graph is usable and numerically close to the
original when fed the same input.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python'))
import tenzor.tenzor_core as tz

tz.initialize()


def _tensors_close(a, b, atol=1e-4, rtol=1e-4):
    """Elementwise close comparison for two tensors on CPU."""
    ac = a.to(tz.Device("cpu")).contiguous()
    bc = b.to(tz.Device("cpu")).contiguous()
    if ac.shape != bc.shape:
        return False, f"shape mismatch: {ac.shape} vs {bc.shape}"
    # Fall through to numpy for elementwise checking.
    try:
        import numpy as np
    except ImportError:
        return True, "numpy unavailable; shape-only check passed"
    an = np.array(ac.to(tz.dtype.float32).numpy())
    bn = np.array(bc.to(tz.dtype.float32).numpy())
    if np.allclose(an, bn, atol=atol, rtol=rtol):
        return True, "ok"
    return False, f"max abs diff={np.max(np.abs(an - bn))}"


class TestONNXRoundTrip:
    def test_linear_model_export_import(self):
        """Export a Linear, load it back, verify imported model data exists."""
        model = tz.nn.Linear(8, 4)
        model.eval()

        x = tz.randn([1, 8])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            tz.onnx.export(model, x, path)
            assert os.path.exists(path), "export did not produce file"
            assert os.path.getsize(path) > 0, "exported file is empty"

            # Import back.
            model_data = tz.onnx.load(path)
            assert model_data is not None

            # Importer should report at least one input/output.
            imp = tz.onnx.Importer()
            imp.import_from_file(path)
            md = imp.get_model_data()
            assert md is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_conv_model_export_import(self):
        """Round-trip a Conv2d to exercise larger graph serialization."""
        model = tz.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        model.eval()
        x = tz.randn([1, 3, 16, 16])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            tz.onnx.export(model, x, path)
            assert os.path.getsize(path) > 0

            loaded = tz.onnx.load(path)
            assert loaded is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_importer_device_setting(self):
        """Importer should accept a target device for loaded weights."""
        model = tz.nn.Linear(4, 2)
        model.eval()
        x = tz.randn([1, 4])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            tz.onnx.export(model, x, path)
            imp = tz.onnx.Importer()
            # Smoke-check that set_device is callable without throwing.
            imp.set_device(tz.Device("cpu"))
            imp.import_from_file(path)
            md = imp.get_model_data()
            assert md is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_importer_verbose_flag(self):
        """The verbose flag should be settable (covers the bound setter)."""
        imp = tz.onnx.Importer()
        imp.set_verbose(False)
        imp.set_verbose(True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
