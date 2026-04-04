"""
Edge case tests for loss functions.

Tests HuberLoss, TripletMarginLoss, and other losses with
various reduction modes, edge cases, and gradient flow.
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


def test_huber_loss_basic():
    _init()
    pred = tz.Variable(tz.randn([4, 3]), True)
    target = tz.Variable(tz.randn([4, 3]), False)

    loss = F.huber_loss(pred, target, delta=1.0, reduction="mean")
    assert loss.shape() == [] or loss.tensor().numel() == 1
    loss.backward()
    assert pred.grad() is not None


def test_huber_loss_reductions():
    _init()
    pred = tz.Variable(tz.randn([4, 3]), True)
    target = tz.Variable(tz.randn([4, 3]), False)

    loss_mean = F.huber_loss(pred, target, reduction="mean")
    loss_sum = F.huber_loss(pred, target, reduction="sum")

    mean_val = float(loss_mean.tensor().item())
    sum_val = float(loss_sum.tensor().item())
    # Sum should be >= mean for positive losses
    assert sum_val >= mean_val or abs(sum_val - mean_val) < 1e-6


def test_smooth_l1_loss_gradient():
    _init()
    pred = tz.Variable(tz.randn([2, 2]), True)
    target = tz.Variable(tz.zeros([2, 2]), False)

    loss = F.smooth_l1_loss(pred, target, beta=0.5)
    loss.backward()
    assert pred.grad() is not None


def test_triplet_margin_loss():
    _init()
    anchor = tz.Variable(tz.randn([4, 8]), True)
    positive = tz.Variable(tz.randn([4, 8]), True)
    negative = tz.Variable(tz.randn([4, 8]), True)

    loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
    val = float(loss.tensor().item())
    assert val >= 0.0, "Triplet loss should be non-negative"

    loss.backward()
    assert anchor.grad() is not None
    assert positive.grad() is not None
    assert negative.grad() is not None


def test_cosine_embedding_loss():
    _init()
    x1 = tz.Variable(tz.randn([4, 8]), True)
    x2 = tz.Variable(tz.randn([4, 8]), True)
    # Target: +1 or -1
    target_data = tz.ones([4])
    target = tz.Variable(target_data, False)

    loss = F.cosine_embedding_loss(x1, x2, target, margin=0.0)
    loss.backward()
    assert x1.grad() is not None


def test_bce_with_logits_gradient():
    _init()
    logits = tz.Variable(tz.randn([4, 1]), True)
    targets = tz.Variable(tz.zeros([4, 1]), False)

    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    assert logits.grad() is not None


def test_kl_div_loss():
    _init()
    log_probs = tz.Variable(tz.randn([4, 5]), True)
    # Target probabilities (positive, summing roughly to 1 per row)
    target = tz.Variable(tz.randn([4, 5]), False)

    loss = F.kl_div(log_probs, target, reduction="mean")
    loss.backward()
    assert log_probs.grad() is not None


def test_mse_loss_zero_input():
    """MSE of identical tensors should be 0."""
    _init()
    x = tz.Variable(tz.randn([3, 3]), True)
    loss = F.mse_loss(x, x)
    val = float(loss.tensor().item())
    assert abs(val) < 1e-6, f"MSE of identical tensors should be ~0, got {val}"


if __name__ == "__main__":
    test_huber_loss_basic()
    test_huber_loss_reductions()
    test_smooth_l1_loss_gradient()
    test_triplet_margin_loss()
    test_cosine_embedding_loss()
    test_bce_with_logits_gradient()
    test_kl_div_loss()
    test_mse_loss_zero_input()
    print("All loss edge case tests passed!")
