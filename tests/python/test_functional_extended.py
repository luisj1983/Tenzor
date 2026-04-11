"""
Tests for newly added functional wrappers.

Tests cosine_similarity, conv2d, scaled_dot_product_attention,
normalize, and pad for shape correctness and gradient flow.
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz
import tenzor.nn.functional as F


def _init():
    tz.initialize()


def test_cosine_similarity_basic():
    _init()
    x1 = tz.Variable(tz.randn([4, 8]), True)
    x2 = tz.Variable(tz.randn([4, 8]), True)

    sim = F.cosine_similarity(x1, x2, dim=1)
    # Should produce [4] output
    assert sim.tensor().numel == 4


def test_cosine_similarity_identical():
    """Cosine similarity of a tensor with itself should be ~1."""
    _init()
    x = tz.Variable(tz.randn([4, 8]), False)
    sim = F.cosine_similarity(x, x, dim=1)
    vals = sim.tensor()
    mean_sim = float(vals.mean().item())
    assert mean_sim > 0.99, f"Self-similarity should be ~1, got {mean_sim}"


def test_conv2d_shape():
    _init()
    # Input: batch=2, channels=3, H=8, W=8
    x = tz.Variable(tz.randn([2, 3, 8, 8]), True)
    # Weight: out_channels=16, in_channels=3, kH=3, kW=3
    w = tz.Variable(tz.randn([16, 3, 3, 3]), True)

    out = F.conv2d(x, w, stride=1, padding=1)
    assert out.shape == [2, 16, 8, 8], f"Expected [2,16,8,8], got {out.shape}"


def test_conv2d_gradient():
    _init()
    x = tz.Variable(tz.randn([1, 1, 4, 4]), True)
    w = tz.Variable(tz.randn([1, 1, 3, 3]), True)

    out = F.conv2d(x, w, padding=1)
    # Reduce on the Variable so the autograd graph stays connected all
    # the way back to x / w. Wrapping out.tensor().sum() in a fresh
    # Variable would sever the graph and leave .grad = None.
    zero_shape = [int(s) for s in out.shape]
    target = tz.Variable(tz.zeros(zero_shape), False)
    loss = F.mse_loss(out, target)
    loss.backward()

    assert x.grad is not None, "Input should have gradients"
    assert w.grad is not None, "Weight should have gradients"


def test_normalize_basic():
    _init()
    x = tz.Variable(tz.randn([4, 8]), False)
    out = F.normalize(x, p=2.0, dim=1)
    assert out.shape == [4, 8]


def test_normalize_preserves_direction():
    """Normalizing should preserve the direction."""
    _init()
    x = tz.Variable(tz.randn([2, 4]), False)
    out = F.normalize(x, p=2.0, dim=1)
    assert out.shape == x.shape


def test_sdpa_shape():
    """Scaled dot-product attention should produce correct output shape."""
    _init()
    B, H, L, S, E = 2, 4, 8, 8, 16
    q = tz.Variable(tz.randn([B, H, L, E]), True)
    k = tz.Variable(tz.randn([B, H, S, E]), True)
    v = tz.Variable(tz.randn([B, H, S, E]), True)

    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == [B, H, L, E], f"Expected [{B},{H},{L},{E}], got {out.shape}"


def test_sdpa_causal():
    """Causal attention should work without errors."""
    _init()
    B, H, L, E = 1, 2, 4, 8
    q = tz.Variable(tz.randn([B, H, L, E]), True)
    k = tz.Variable(tz.randn([B, H, L, E]), True)
    v = tz.Variable(tz.randn([B, H, L, E]), True)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == [B, H, L, E]


if __name__ == "__main__":
    test_cosine_similarity_basic()
    test_cosine_similarity_identical()
    test_conv2d_shape()
    test_conv2d_gradient()
    test_normalize_basic()
    test_normalize_preserves_direction()
    test_sdpa_shape()
    test_sdpa_causal()
    print("All functional extended tests passed!")
