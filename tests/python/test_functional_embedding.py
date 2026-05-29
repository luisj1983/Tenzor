"""
Functional embedding binding correctness (release-prep A1).

Regression for a P0 where ``nn.functional_embedding`` constructed a fresh
randomly-initialised Embedding layer and ignored the caller's ``weight`` — so
the forward returned lookups into garbage and the gradient never reached the
supplied weight. The fix routes the binding through
``tenzor::nn::functional::embedding(input, weight, padding_idx)`` which is
autograd-aware w.r.t. the supplied weight.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init_tenzor():
    tz.initialize()
    tz.manual_seed(1234)


def _scalar(v):
    t = v.tensor() if hasattr(v, "tensor") else v
    return float(t.item())


def test_functional_embedding_uses_supplied_weight():
    num_emb, dim = 6, 3
    # Constant weight: every looked-up element must be exactly 2.5 if (and only
    # if) the supplied weight is actually used. A fresh random layer would not.
    weight = tz.Variable(tz.full([num_emb, dim], 2.5), True)
    idx = tz.Variable(tz.tensor([0, 1, 2, 3], tz.dtype.int64), False)

    out = tz.nn.functional_embedding(idx, weight, -1)
    assert abs(_scalar(tz.sum(out)) - (4 * dim * 2.5)) < 1e-4

    tz.sum(out).backward()
    assert weight.grad is not None, "gradient did not flow to the supplied weight"
    # 4 distinct rows each receive grad 1 across `dim` columns.
    assert abs(_scalar(weight.grad.sum()) - (4 * dim)) < 1e-4


def test_functional_embedding_padding_idx_zeros_grad():
    num_emb, dim = 6, 3
    idx = tz.Variable(tz.tensor([0, 1, 2, 3], tz.dtype.int64), False)

    w_all = tz.Variable(tz.full([num_emb, dim], 2.5), True)
    tz.sum(tz.nn.functional_embedding(idx, w_all, -1)).backward()
    g_all = _scalar(w_all.grad.sum())

    w_pad = tz.Variable(tz.full([num_emb, dim], 2.5), True)
    tz.sum(tz.nn.functional_embedding(idx, w_pad, 1)).backward()  # padding_idx=1
    g_pad = _scalar(w_pad.grad.sum())

    assert abs(g_all - 4 * dim) < 1e-4
    # padding_idx=1 must zero row 1's gradient -> one fewer row contributes.
    assert abs(g_pad - 3 * dim) < 1e-4
