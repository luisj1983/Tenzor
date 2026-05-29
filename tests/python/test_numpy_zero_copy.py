"""
True zero-copy NumPy import (release-prep B9).

Tensor.from_numpy now wraps a C-contiguous CPU NumPy buffer without copying
(sharing memory, torch.from_numpy semantics) instead of always copying.
"""
import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module", autouse=True)
def _init():
    tz.initialize()


def test_from_numpy_zero_copy_shares_memory():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t = tz.Tensor.from_numpy(a)
    a[0] = 99.0  # mutate the numpy array after wrapping
    # Zero-copy => the tensor sees the mutation (shared buffer).
    assert float(t[0].item()) == 99.0


def test_from_numpy_noncontiguous_copies():
    base = np.arange(8, dtype=np.float32).reshape(2, 4)
    view = base[:, ::2]  # non-contiguous strided view
    t = tz.Tensor.from_numpy(view)
    base[0, 0] = 77.0  # mutate original
    # Non-contiguous source is copied => mutation NOT reflected.
    assert float(t[0, 0].item()) != 77.0
