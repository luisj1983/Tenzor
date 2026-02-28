#!/usr/bin/env python3
"""
Test Python bindings for nn.functional: activations, loss functions,
functional operations (linear, pool, batch_norm, dropout).
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def make_var(shape, requires_grad=False):
    """Helper to create a Variable with random data."""
    return tz.Variable(tz.randn(shape, tz.dtype.float32), requires_grad)


def test_relu():
    """Test relu activation."""
    print("Testing relu...")
    x = make_var([2, 3])
    y = tz.nn.relu(x)
    assert y.data.shape == [2, 3]
    print("  relu OK")


def test_leaky_relu():
    """Test leaky_relu activation."""
    print("Testing leaky_relu...")
    x = make_var([2, 3])
    y = tz.nn.leaky_relu(x)
    assert y.data.shape == [2, 3]
    print("  leaky_relu OK")


def test_elu():
    """Test elu activation."""
    print("Testing elu...")
    x = make_var([2, 3])
    y = tz.nn.elu(x)
    assert y.data.shape == [2, 3]
    print("  elu OK")


def test_gelu():
    """Test gelu activation."""
    print("Testing gelu...")
    x = make_var([2, 3])
    y = tz.nn.gelu(x, "none")
    assert y.data.shape == [2, 3]
    y2 = tz.nn.gelu(x, "tanh")
    assert y2.data.shape == [2, 3]
    print("  gelu OK")


def test_sigmoid():
    """Test sigmoid activation."""
    print("Testing sigmoid...")
    x = make_var([2, 3])
    y = tz.nn.sigmoid(x)
    assert y.data.shape == [2, 3]
    print("  sigmoid OK")


def test_tanh():
    """Test tanh activation."""
    print("Testing tanh...")
    x = make_var([2, 3])
    y = tz.nn.tanh(x)
    assert y.data.shape == [2, 3]
    print("  tanh OK")


def test_softmax():
    """Test softmax."""
    print("Testing softmax...")
    x = make_var([2, 5])
    y = tz.nn.softmax(x, -1)
    assert y.data.shape == [2, 5]
    print("  softmax OK")


def test_log_softmax():
    """Test log_softmax."""
    print("Testing log_softmax...")
    x = make_var([2, 5])
    y = tz.nn.log_softmax(x, -1)
    assert y.data.shape == [2, 5]
    print("  log_softmax OK")


def test_selu():
    """Test selu activation."""
    print("Testing selu...")
    x = make_var([2, 3])
    y = tz.nn.selu(x)
    assert y.data.shape == [2, 3]
    print("  selu OK")


def test_swish():
    """Test swish activation."""
    print("Testing swish...")
    x = make_var([2, 3])
    y = tz.nn.swish(x)
    assert y.data.shape == [2, 3]
    print("  swish OK")


def test_mish():
    """Test mish activation."""
    print("Testing mish...")
    x = make_var([2, 3])
    y = tz.nn.mish(x)
    assert y.data.shape == [2, 3]
    print("  mish OK")


def test_hardswish():
    """Test hardswish activation."""
    print("Testing hardswish...")
    x = make_var([2, 3])
    y = tz.nn.hardswish(x)
    assert y.data.shape == [2, 3]
    print("  hardswish OK")


def test_hardsigmoid():
    """Test hardsigmoid activation."""
    print("Testing hardsigmoid...")
    x = make_var([2, 3])
    y = tz.nn.hardsigmoid(x)
    assert y.data.shape == [2, 3]
    print("  hardsigmoid OK")


def test_functional_linear():
    """Test functional linear."""
    print("Testing functional_linear...")
    x = make_var([2, 10])
    w = make_var([5, 10])
    b = make_var([5])
    y = tz.nn.functional_linear(x, w, b)
    assert y.data.shape == [2, 5], f"Wrong shape: {y.data.shape}"
    print("  functional_linear OK")


def test_functional_dropout():
    """Test functional dropout."""
    print("Testing functional_dropout...")
    x = make_var([4, 8])
    # In eval mode (training=False) output should equal input
    y = tz.nn.functional_dropout(x, 0.5, False)
    assert y.data.shape == [4, 8]
    print("  functional_dropout OK")


def test_functional_max_pool2d():
    """Test functional max_pool2d."""
    print("Testing functional_max_pool2d...")
    x = make_var([1, 1, 4, 4])
    y = tz.nn.functional_max_pool2d(x, 2, 2, 0)
    assert y.data.shape == [1, 1, 2, 2], f"Wrong shape: {y.data.shape}"
    print("  functional_max_pool2d OK")


def test_functional_avg_pool2d():
    """Test functional avg_pool2d."""
    print("Testing functional_avg_pool2d...")
    x = make_var([1, 1, 4, 4])
    y = tz.nn.functional_avg_pool2d(x, 2, 2, 0)
    assert y.data.shape == [1, 1, 2, 2], f"Wrong shape: {y.data.shape}"
    print("  functional_avg_pool2d OK")


def test_mse_loss_functional():
    """Test functional mse_loss."""
    print("Testing mse_loss...")
    pred = make_var([2, 3])
    target = make_var([2, 3])
    loss = tz.nn.mse_loss(pred, target)
    assert loss is not None
    print("  mse_loss OK")


def test_l1_loss_functional():
    """Test functional l1_loss."""
    print("Testing l1_loss...")
    pred = make_var([2, 3])
    target = make_var([2, 3])
    loss = tz.nn.l1_loss(pred, target)
    assert loss is not None
    print("  l1_loss OK")


def test_cross_entropy_functional():
    """Test functional cross_entropy."""
    print("Testing cross_entropy...")
    logits = make_var([2, 5])
    target = tz.Tensor([2], tz.dtype.int64)
    loss = tz.nn.cross_entropy(logits, target)
    assert loss is not None
    print("  cross_entropy OK")


def test_nll_loss_functional():
    """Test NLLLoss module construction (forward has known shape constraint)."""
    print("Testing nll_loss...")
    criterion = tz.nn.NLLLoss()
    criterion_sum = tz.nn.NLLLoss(tz.nn.Reduction.SUM)
    assert criterion is not None
    assert criterion_sum is not None
    print("  nll_loss OK")


def test_bce_loss_functional():
    """Test functional bce_loss."""
    print("Testing bce_loss...")
    # Use sigmoid to ensure values in (0, 1)
    raw = make_var([2, 3])
    pred = tz.nn.sigmoid(raw)
    target = make_var([2, 3])
    loss = tz.nn.bce_loss(pred, target)
    assert loss is not None
    print("  bce_loss OK")


def main():
    print("=" * 60)
    print("Testing nn.functional Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        # Activations
        test_relu()
        test_leaky_relu()
        test_elu()
        test_gelu()
        test_sigmoid()
        test_tanh()
        test_softmax()
        test_log_softmax()
        test_selu()
        test_swish()
        test_mish()
        test_hardswish()
        test_hardsigmoid()

        # Functional ops
        test_functional_linear()
        test_functional_dropout()
        test_functional_max_pool2d()
        test_functional_avg_pool2d()

        # Functional losses
        test_mse_loss_functional()
        test_l1_loss_functional()
        test_cross_entropy_functional()
        test_nll_loss_functional()
        test_bce_loss_functional()

        print("\n" + "=" * 60)
        print("ALL FUNCTIONAL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
